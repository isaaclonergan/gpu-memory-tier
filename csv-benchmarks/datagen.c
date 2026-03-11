#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define BUFFER_SIZE 1024

float rand_float(float min, float max) {
    return min + (max - min) * ((float)rand() / RAND_MAX);
}

// Generate a row of sensor data
void generate_sensor_row(FILE *f, double timestamp) {
    float altitude = rand_float(0, 12000);
    float airspeed = rand_float(0, 250);
    float pitch = rand_float(-10, 10);  // small realistic variations
    float roll  = rand_float(-20, 20);
    float yaw   = rand_float(0, 360);
    float temp  = rand_float(-50, 50);
    float pressure = rand_float(950, 1050);
    float engine_rpm = rand_float(0, 5000);

    fprintf(f, "%.2f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
            timestamp, altitude, airspeed, pitch, roll, yaw, temp, pressure);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <output_file.csv> <target_size_MB>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    double target_size_MB = atof(argv[2]);
    if (target_size_MB <= 0) {
        printf("Invalid target size.\n");
        return 1;
    }

    FILE *f = fopen(filename, "w");
    if (!f) { perror("fopen"); return 1; }

    srand(time(NULL));

    fprintf(f, "timestamp,altitude,airspeed,pitch,roll,yaw,temp,pressure\n");

    size_t target_bytes = (size_t)(target_size_MB * 1024 * 1024);
    size_t written_bytes = ftell(f);
    double timestamp = 0.0;
    double timestep = 0.1; // 10 Hz

    char buffer[BUFFER_SIZE];

    while (written_bytes < target_bytes) {
        // Generate row in buffer first
        int len = snprintf(buffer, BUFFER_SIZE, "%.2f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
                           timestamp,
                           rand_float(0,12000),
                           rand_float(0,250),
                           rand_float(-10,10),
                           rand_float(-20,20),
                           rand_float(0,360),
                           rand_float(-50,50),
                           rand_float(950,1050));
        fwrite(buffer, 1, len, f);
        written_bytes += len;
        timestamp += timestep;
    }

    fclose(f);
    printf("Finished generating ~%.2f MB of sensor data into '%s'\n", (double)written_bytes/(1024*1024), filename);
    return 0;
}
