#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <cstdint>
#include <cstring>

#define PAGE_SIZE       (64   * 1024)
#define CPU_CAP         (2ULL * 1024 * 1024 * 1024)
#define BATCH_CAP       (512ULL * 1024 * 1024)
#define PINNED_BUF_SIZE (512ULL * 1024 * 1024)

struct SensorRecord
{
    float timestamp;
    float altitude;
    float airspeed;
    float pitch;
    float roll;
    float yaw;
    float temp;
    float pressure;
};

struct Page
{
    void*    cpu_ptr;
    void*    gpu_ptr;
    size_t   size;
    size_t   id;
    bool     on_gpu;
    bool     sorted;
    size_t   num_records;
};

struct TierStats
{
    size_t   cpu_used;
    size_t   gpu_used;
    size_t   total_migrations_to_cpu;
    size_t   total_migrations_to_gpu;
    size_t   total_bytes_transferred;
    double   total_transfer_time;
    double   total_sort_time;

    TierStats() : cpu_used(0), gpu_used(0),
                  total_migrations_to_cpu(0),
                  total_migrations_to_gpu(0),
                  total_bytes_transferred(0),
                  total_transfer_time(0),
                  total_sort_time(0) {}
};

struct PageTable
{
    std::vector<Page> pages;
    TierStats         stats;
    std::mutex        lock;
    void*             pinned_buf_a;
    void*             pinned_buf_b;
    cudaStream_t      stream_a;
    cudaStream_t      stream_b;

    PageTable() : pinned_buf_a(nullptr), pinned_buf_b(nullptr) {}
};

size_t records_per_page()
{
    return PAGE_SIZE / sizeof(SensorRecord);
}

bool init_resources(PageTable& table)
{
    cudaMallocHost(&table.pinned_buf_a, PINNED_BUF_SIZE);
    cudaMallocHost(&table.pinned_buf_b, PINNED_BUF_SIZE);
    cudaStreamCreate(&table.stream_a);
    cudaStreamCreate(&table.stream_b);

    std::cout << "GPU far memory tier ready\n";
    std::cout << "Pinned buffers : 2 x "
              << PINNED_BUF_SIZE / (1024*1024) << " MB\n\n";
    return true;
}

void cleanup(PageTable& table)
{
    if (table.pinned_buf_a) cudaFreeHost(table.pinned_buf_a);
    if (table.pinned_buf_b) cudaFreeHost(table.pinned_buf_b);
    cudaStreamDestroy(table.stream_a);
    cudaStreamDestroy(table.stream_b);
    for (auto& p : table.pages)
    {
        if (p.cpu_ptr) free(p.cpu_ptr);
        if (p.gpu_ptr) cudaFree(p.gpu_ptr);
    }
}

void free_page_from_cpu(PageTable& table, Page& page)
{
    if (page.cpu_ptr)
    {
        free(page.cpu_ptr);
        page.cpu_ptr          = nullptr;
        table.stats.cpu_used -= page.size;
    }
}

void fetch_batch_from_gpu(PageTable&           table,
                           std::vector<size_t>& page_ids,
                           cudaStream_t         stream,
                           void*                pinned_buf)
{
    std::vector<size_t> gpu_ids;
    for (size_t idx : page_ids)
        if (table.pages[idx].on_gpu && table.pages[idx].gpu_ptr)
            gpu_ids.push_back(idx);

    if (gpu_ids.empty()) return;

    auto t_start = std::chrono::high_resolution_clock::now();

    size_t pages_per_pinned = PINNED_BUF_SIZE / PAGE_SIZE;
    size_t chunk_start      = 0;

    while (chunk_start < gpu_ids.size())
    {
        size_t chunk_end = std::min(chunk_start + pages_per_pinned,
                                    gpu_ids.size());

        for (size_t i = chunk_start; i < chunk_end; i++)
        {
            size_t idx    = gpu_ids[i];
            Page&  p      = table.pages[idx];
            size_t offset = (i - chunk_start) * PAGE_SIZE;

            cudaMemcpyAsync((char*)pinned_buf + offset,
                             p.gpu_ptr,
                             p.size,
                             cudaMemcpyDeviceToHost,
                             stream);
        }

        cudaStreamSynchronize(stream);

        for (size_t i = chunk_start; i < chunk_end; i++)
        {
            size_t idx    = gpu_ids[i];
            Page&  p      = table.pages[idx];
            size_t offset = (i - chunk_start) * PAGE_SIZE;

            p.cpu_ptr = malloc(p.size);
            memcpy(p.cpu_ptr,
                   (char*)pinned_buf + offset,
                   p.size);

            cudaFree(p.gpu_ptr);
            p.gpu_ptr                           = nullptr;
            p.on_gpu                            = false;
            table.stats.gpu_used               -= p.size;
            table.stats.cpu_used               += p.size;
            table.stats.total_migrations_to_cpu++;
            table.stats.total_bytes_transferred += p.size;
        }

        chunk_start = chunk_end;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t_end - t_start).count();
    table.stats.total_transfer_time += secs;

    double mb = (gpu_ids.size() * PAGE_SIZE) / (1024.0 * 1024.0);
    double bw = (mb / 1024.0) / secs;

    std::cout << "  Fetched " << gpu_ids.size() << " pages ("
              << (size_t)mb << " MB) from GPU"
              << " in " << secs << "s"
              << " @ " << bw << " GB/s\n";
}

void load_csv_into_pages(const std::string& filename,
                         PageTable& table)
{
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line);

    std::cout << "Loading: " << filename << "\n";

    const size_t RECS = records_per_page();

    void* pinned_stage = nullptr;
    cudaMallocHost(&pinned_stage, PAGE_SIZE);

    std::vector<SensorRecord> buffer;
    buffer.reserve(RECS);

    size_t total_records = 0;
    size_t page_id       = 0;

    auto flush_page = [&]()
    {
        if (buffer.empty()) return;

        Page p;
        p.size        = PAGE_SIZE;
        p.id          = page_id++;
        p.on_gpu      = false;
        p.sorted      = false;
        p.num_records = buffer.size();
        p.cpu_ptr     = nullptr;
        p.gpu_ptr     = nullptr;

        std::lock_guard<std::mutex> guard(table.lock);

        if (table.stats.cpu_used + PAGE_SIZE <= CPU_CAP)
        {
            p.cpu_ptr = malloc(PAGE_SIZE);
            memset(p.cpu_ptr, 0, PAGE_SIZE);
            memcpy(p.cpu_ptr, buffer.data(),
                   buffer.size() * sizeof(SensorRecord));
            table.stats.cpu_used += PAGE_SIZE;
        }
        else
        {
            memset(pinned_stage, 0, PAGE_SIZE);
            memcpy(pinned_stage, buffer.data(),
                   buffer.size() * sizeof(SensorRecord));

            cudaMalloc(&p.gpu_ptr, PAGE_SIZE);
            cudaMemcpy(p.gpu_ptr, pinned_stage,
                       PAGE_SIZE, cudaMemcpyHostToDevice);

            p.on_gpu = true;
            table.stats.gpu_used               += PAGE_SIZE;
            table.stats.total_migrations_to_gpu++;
            table.stats.total_bytes_transferred += PAGE_SIZE;
        }

        table.pages.push_back(p);
        buffer.clear();

        if (page_id % 5000 == 0)
            std::cout << "  " << page_id << " pages | "
                      << "CPU: " << table.stats.cpu_used/(1024*1024)
                      << " MB GPU: " << table.stats.gpu_used/(1024*1024)
                      << " MB\n";
    };

    while (std::getline(file, line))
    {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string field;
        SensorRecord rec;

        std::getline(ss, field, ','); rec.timestamp = std::stof(field);
        std::getline(ss, field, ','); rec.altitude   = std::stof(field);
        std::getline(ss, field, ','); rec.airspeed   = std::stof(field);
        std::getline(ss, field, ','); rec.pitch      = std::stof(field);
        std::getline(ss, field, ','); rec.roll       = std::stof(field);
        std::getline(ss, field, ','); rec.yaw        = std::stof(field);
        std::getline(ss, field, ','); rec.temp       = std::stof(field);
        std::getline(ss, field, ','); rec.pressure   = std::stof(field);

        buffer.push_back(rec);
        total_records++;

        if (buffer.size() == RECS) flush_page();
    }

    flush_page();
    cudaFreeHost(pinned_stage);
    file.close();

    std::cout << "Load complete: " << total_records
              << " records in " << page_id << " pages\n";
}

void sort_batched(PageTable& table)
{
    size_t total_pages     = table.pages.size();
    size_t pages_per_batch = BATCH_CAP / PAGE_SIZE;
    size_t buf_records     = pages_per_batch * records_per_page();

    std::cout << "\n========== Sort Phase ==========\n";
    std::cout << "Total pages     : " << total_pages     << "\n";
    std::cout << "Pages per batch : " << pages_per_batch << "\n\n";

    SensorRecord* batch_array =
        (SensorRecord*)malloc(buf_records * sizeof(SensorRecord));

    size_t page_idx     = 0;
    size_t batch_num    = 0;
    bool   use_stream_a = true;

    while (page_idx < total_pages)
    {
        batch_num++;

        std::vector<size_t> batch_ids;
        size_t batch_bytes = 0;

        for (size_t i = page_idx;
             i < total_pages && batch_bytes + PAGE_SIZE <= BATCH_CAP;
             i++)
        {
            if (!table.pages[i].sorted)
            {
                batch_ids.push_back(i);
                batch_bytes += PAGE_SIZE;
            }
        }

        if (batch_ids.empty()) break;

        page_idx = batch_ids.back() + 1;

        std::cout << "--- Batch " << batch_num
                  << " | " << batch_ids.size() << " pages ("
                  << batch_bytes / (1024*1024) << " MB) ---\n";

        cudaStream_t stream     = use_stream_a
                                  ? table.stream_a : table.stream_b;
        void*        pinned_buf = use_stream_a
                                  ? table.pinned_buf_a : table.pinned_buf_b;

        fetch_batch_from_gpu(table, batch_ids, stream, pinned_buf);

        std::cout << "  CPU after fetch : "
                  << table.stats.cpu_used / (1024*1024) << " MB\n";
        std::cout << "  GPU after fetch : "
                  << table.stats.gpu_used / (1024*1024) << " MB\n";

        size_t write_pos = 0;

        for (size_t idx : batch_ids)
        {
            Page& p = table.pages[idx];
            if (!p.cpu_ptr) continue;

            memcpy(batch_array + write_pos,
                   (SensorRecord*)p.cpu_ptr,
                   p.num_records * sizeof(SensorRecord));

            write_pos += p.num_records;
            free_page_from_cpu(table, p);
        }

        std::cout << "  Records extracted : " << write_pos << "\n";
        std::cout << "  CPU after extract : "
                  << table.stats.cpu_used / (1024*1024) << " MB\n";

        auto t2 = std::chrono::high_resolution_clock::now();

        std::sort(batch_array, batch_array + write_pos,
                  [](const SensorRecord& a, const SensorRecord& b)
                  { return a.altitude < b.altitude; });

        auto t3 = std::chrono::high_resolution_clock::now();

        double sort_sec = std::chrono::duration<double>(t3-t2).count();
        table.stats.total_sort_time += sort_sec;

        float last_max = batch_array[write_pos-1].altitude;

        std::cout << "  Sort time        : " << sort_sec << "s\n";
        std::cout << "  Lowest altitude  : "
                  << batch_array[0].altitude << "\n";
        std::cout << "  Highest altitude : " << last_max << "\n";

        for (size_t idx : batch_ids)
            table.pages[idx].sorted = true;

        use_stream_a = !use_stream_a;

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cout << "  GPU (nvidia-smi) : "
                  << (total_mem - free_mem) / (1024*1024) << " MB\n";
    }

    free(batch_array);

    std::cout << "\n========================================\n";
    std::cout << "Sort complete\n";
    std::cout << "========================================\n";
}

void print_results(PageTable& table,
                   double load_sec,
                   double sort_sec)
{
    TierStats& s = table.stats;

    double avg_bw = s.total_transfer_time > 0
        ? (s.total_bytes_transferred / (1024.0*1024.0*1024.0))
          / s.total_transfer_time
        : 0;

    std::cout << "\n========== GPU Tier Benchmark ==========\n";
    std::cout << "Load time             : " << load_sec              << "s\n";
    std::cout << "Sort time (total)     : " << s.total_sort_time     << "s\n";
    std::cout << "Transfer time (total) : " << s.total_transfer_time << "s\n";
    std::cout << "Sort+Transfer total   : " << sort_sec              << "s\n";
    std::cout << "\nGPU far memory bandwidth\n";
    std::cout << "  Bytes transferred   : "
              << s.total_bytes_transferred / (1024*1024)        << " MB\n";
    std::cout << "  Avg transfer BW     : " << avg_bw           << " GB/s\n";
    std::cout << "  GPU->CPU migrations : " << s.total_migrations_to_cpu  << "\n";
    std::cout << "  CPU->GPU migrations : " << s.total_migrations_to_gpu  << "\n";
    std::cout << "========================================\n";
}

int main()
{
    const std::string INPUT_FILE = "dataset.csv";

    PageTable table;

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "GPU free  : " << free_mem  / (1024*1024) << " MB\n";
    std::cout << "GPU total : " << total_mem / (1024*1024) << " MB\n\n";

    init_resources(table);

    auto t0 = std::chrono::high_resolution_clock::now();
    load_csv_into_pages(INPUT_FILE, table);
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_sec = std::chrono::duration<double>(t1-t0).count();

    std::cout << "Load time: " << load_sec << "s\n\n";

    auto t2 = std::chrono::high_resolution_clock::now();
    sort_batched(table);
    auto t3 = std::chrono::high_resolution_clock::now();
    double sort_sec = std::chrono::duration<double>(t3-t2).count();

    print_results(table, load_sec, sort_sec);

    cleanup(table);
    return 0;
}
