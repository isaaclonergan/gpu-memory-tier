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
#define SWAP_READ_BUF   (512ULL * 1024 * 1024)

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
    size_t   size;
    size_t   id;
    bool     on_swap;
    bool     sorted;
    size_t   num_records;
    size_t   swap_offset;
};

struct TierStats
{
    size_t   cpu_used;
    size_t   swap_used;
    size_t   total_migrations_to_cpu;
    size_t   total_migrations_to_swap;
    size_t   total_bytes_transferred;
    double   total_transfer_time;
    double   total_sort_time;

    TierStats() : cpu_used(0), swap_used(0),
                  total_migrations_to_cpu(0),
                  total_migrations_to_swap(0),
                  total_bytes_transferred(0),
                  total_transfer_time(0),
                  total_sort_time(0) {}
};

struct PageTable
{
    std::vector<Page> pages;
    TierStats         stats;
    std::mutex        lock;
    FILE*             swap_file;
    std::string       swap_filename;
    void*             swap_read_buf;

    PageTable() : swap_file(nullptr), swap_read_buf(nullptr) {}
};

size_t records_per_page()
{
    return PAGE_SIZE / sizeof(SensorRecord);
}

bool init_resources(PageTable& table, const std::string& swap_filename)
{
    table.swap_filename = swap_filename;
    table.swap_file     = fopen(swap_filename.c_str(), "w+b");

    if (!table.swap_file)
    {
        std::cerr << "Failed to open swap file\n";
        return false;
    }

    table.swap_read_buf = malloc(SWAP_READ_BUF);

    std::cout << "Swap far memory tier ready\n";
    std::cout << "Swap file   : " << swap_filename << "\n";
    std::cout << "Read buffer : "
              << SWAP_READ_BUF / (1024*1024) << " MB\n\n";
    return true;
}

void cleanup(PageTable& table)
{
    if (table.swap_file)     fclose(table.swap_file);
    if (table.swap_read_buf) free(table.swap_read_buf);
    remove(table.swap_filename.c_str());
    for (auto& p : table.pages)
        if (p.cpu_ptr) free(p.cpu_ptr);
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

void fetch_batch_from_swap(PageTable&           table,
                            std::vector<size_t>& page_ids)
{
    std::vector<size_t> swap_ids;
    for (size_t idx : page_ids)
        if (table.pages[idx].on_swap)
            swap_ids.push_back(idx);

    if (swap_ids.empty()) return;

    auto t_start = std::chrono::high_resolution_clock::now();

    size_t pages_per_buf = SWAP_READ_BUF / PAGE_SIZE;
    size_t chunk_start   = 0;

    while (chunk_start < swap_ids.size())
    {
        size_t chunk_end = std::min(chunk_start + pages_per_buf,
                                    swap_ids.size());

        // check contiguous
        bool contiguous = true;
        for (size_t i = chunk_start + 1; i < chunk_end; i++)
        {
            if (table.pages[swap_ids[i]].swap_offset !=
                table.pages[swap_ids[i-1]].swap_offset + PAGE_SIZE)
            {
                contiguous = false;
                break;
            }
        }

        if (contiguous)
        {
            size_t first_idx   = swap_ids[chunk_start];
            size_t total_bytes = (chunk_end - chunk_start) * PAGE_SIZE;

            fseek(table.swap_file,
                  (long)table.pages[first_idx].swap_offset,
                  SEEK_SET);

            fread(table.swap_read_buf, 1, total_bytes,
                  table.swap_file);

            for (size_t i = chunk_start; i < chunk_end; i++)
            {
                size_t idx    = swap_ids[i];
                Page&  p      = table.pages[idx];
                size_t offset = (i - chunk_start) * PAGE_SIZE;

                p.cpu_ptr = malloc(p.size);
                memcpy(p.cpu_ptr,
                       (char*)table.swap_read_buf + offset,
                       p.size);

                p.on_swap                           = false;
                table.stats.swap_used              -= p.size;
                table.stats.cpu_used               += p.size;
                table.stats.total_migrations_to_cpu++;
                table.stats.total_bytes_transferred += p.size;
            }
        }
        else
        {
            for (size_t i = chunk_start; i < chunk_end; i++)
            {
                size_t idx = swap_ids[i];
                Page&  p   = table.pages[idx];

                p.cpu_ptr = malloc(p.size);
                fseek(table.swap_file, (long)p.swap_offset, SEEK_SET);
                fread(p.cpu_ptr, 1, p.size, table.swap_file);

                p.on_swap                           = false;
                table.stats.swap_used              -= p.size;
                table.stats.cpu_used               += p.size;
                table.stats.total_migrations_to_cpu++;
                table.stats.total_bytes_transferred += p.size;
            }
        }

        chunk_start = chunk_end;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t_end - t_start).count();
    table.stats.total_transfer_time += secs;

    double mb = (swap_ids.size() * PAGE_SIZE) / (1024.0 * 1024.0);
    double bw = (mb / 1024.0) / secs;

    std::cout << "  Fetched " << swap_ids.size() << " pages ("
              << (size_t)mb << " MB) from swap"
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
        p.on_swap     = false;
        p.sorted      = false;
        p.num_records = buffer.size();
        p.cpu_ptr     = nullptr;
        p.swap_offset = 0;

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
            void* tmp = malloc(PAGE_SIZE);
            memset(tmp, 0, PAGE_SIZE);
            memcpy(tmp, buffer.data(),
                   buffer.size() * sizeof(SensorRecord));

            fseek(table.swap_file, 0, SEEK_END);
            p.swap_offset = (size_t)ftell(table.swap_file);
            fwrite(tmp, 1, PAGE_SIZE, table.swap_file);
            fflush(table.swap_file);
            free(tmp);

            p.on_swap  = true;
            table.stats.swap_used              += PAGE_SIZE;
            table.stats.total_migrations_to_swap++;
            table.stats.total_bytes_transferred += PAGE_SIZE;
        }

        table.pages.push_back(p);
        buffer.clear();

        if (page_id % 5000 == 0)
            std::cout << "  " << page_id << " pages | "
                      << "CPU: " << table.stats.cpu_used/(1024*1024)
                      << " MB Swap: " << table.stats.swap_used/(1024*1024)
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

    size_t page_idx  = 0;
    size_t batch_num = 0;
    float  last_max  = -1e38f;

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

        auto t0 = std::chrono::high_resolution_clock::now();
        fetch_batch_from_swap(table, batch_ids);
        auto t1 = std::chrono::high_resolution_clock::now();

        std::cout << "  CPU after fetch : "
                  << table.stats.cpu_used  / (1024*1024) << " MB\n";
        std::cout << "  Swap after fetch: "
                  << table.stats.swap_used / (1024*1024) << " MB\n";

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

        last_max = batch_array[write_pos-1].altitude;

        std::cout << "  Sort time        : " << sort_sec  << "s\n";
        std::cout << "  Lowest altitude  : "
                  << batch_array[0].altitude   << "\n";
        std::cout << "  Highest altitude : " << last_max  << "\n";

        for (size_t idx : batch_ids)
            table.pages[idx].sorted = true;
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

    std::cout << "\n========== Swap Tier Benchmark ==========\n";
    std::cout << "Load time             : " << load_sec              << "s\n";
    std::cout << "Sort time (total)     : " << s.total_sort_time     << "s\n";
    std::cout << "Transfer time (total) : " << s.total_transfer_time << "s\n";
    std::cout << "Sort+Transfer total   : " << sort_sec              << "s\n";
    std::cout << "\nSwap far memory bandwidth\n";
    std::cout << "  Bytes transferred    : "
              << s.total_bytes_transferred / (1024*1024)              << " MB\n";
    std::cout << "  Avg transfer BW      : " << avg_bw                << " GB/s\n";
    std::cout << "  Swap->CPU migrations : " << s.total_migrations_to_cpu  << "\n";
    std::cout << "  CPU->Swap migrations : " << s.total_migrations_to_swap << "\n";
    std::cout << "=========================================\n";
}

int main()
{
    const std::string INPUT_FILE = "dataset.csv";
    const std::string SWAP_FILE  = "page_swap.bin";

    PageTable table;

    init_resources(table, SWAP_FILE);

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
