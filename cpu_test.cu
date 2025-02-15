#include <stdio.h>
#include <time.h>

int main()
{
    int size = 1 << 30; // 2^20
    int *arr;
    arr = (int *)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++)
    {
        arr[i] = i;
    }
    int result = -100000000;
    clock_t start_time = clock();
    for (int i = 0; i < size; i++)
    {
        if (arr[i] > result)
            result = arr[i];
    }

    clock_t end_time = clock();

    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Time taken: %f ms\n", time_taken * pow(10, 3));

    return 0;
}