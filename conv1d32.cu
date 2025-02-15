#include <iostream>

using namespace std;

__global__ void conv1D(int *signal, int *filter, int radius, int signal_length, int *output)
{
    int idx = threadIdx.x;
    int value = 0;
    for (int i = -radius; i <= radius; i++)
    {
        if ((idx + i) >= 0 && (idx + i) < signal_length)
        {
            value += signal[idx + i] * filter[radius + i];
        }
    }
    output[threadIdx.x] = value;
}

int main(void)
{
    int filter_length;
    int input_signal_length;

    cout << "Enter filter length" << endl;
    cin >> filter_length;
    int h_filter[filter_length];

    cout << "Enter input signal length" << endl;
    cin >> input_signal_length;
    int h_input_signal[input_signal_length];

    cout << "Enter filter elements seperated by space" << endl;
    for (int i = 0; i < filter_length; i++)
    {
        cin >> h_filter[i];
    }

    cout << "Enter signal elements seperated by space" << endl;
    for (int i = 0; i < input_signal_length; i++)
    {
        cin >> h_input_signal[i];
    }

    int *d_filter;
    int *d_input_signal;
    int *d_output;
    cudaMalloc((void **)&d_filter, filter_length * sizeof(int));
    cudaMalloc((void **)&d_input_signal, input_signal_length * sizeof(int));
    cudaMalloc((void **)&d_output, input_signal_length * sizeof(int));
    cudaMemcpy(d_filter, h_filter, filter_length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_signal, h_input_signal, input_signal_length * sizeof(int), cudaMemcpyHostToDevice);

    cout << "radius " << filter_length / 2 << endl;
    cout << "input_signal_length " << input_signal_length << endl;

    conv1D<<<1, input_signal_length>>>(d_input_signal, d_filter, filter_length / 2, input_signal_length, d_output);

    int h_output[input_signal_length];
    cudaMemcpy(h_output, d_output, (input_signal_length) * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Output is" << endl;
    for (int i = 0; i < input_signal_length; i++)
    {
        cout << h_output[i] << " ";
    }
    cout << endl;

    return 0;
}