#include <iostream>
#include <vector>
#include <complex>
#include <fftw3.h>

using namespace std;

class FourierTransform {
public:
    FourierTransform(int size) : size(size) {
        inData = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * size);
        outData = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * size);

        forwardPlan = fftw_plan_dft_1d(size, inData, outData, FFTW_FORWARD, FFTW_ESTIMATE);
        inversePlan = fftw_plan_dft_1d(size, outData, inData, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    vector<complex<double>> forward(vector<complex<double>>& input) {
        if (input.size() != size) {
            cerr << "Ошибка: Размер входных данных должен соответствовать размеру FFT." << endl;
            return vector<complex<double>>();
        }

        for (int i = 0; i < size; i++) {
            inData[i][0] = input[i].real();
            inData[i][1] = input[i].imag();
        }

        fftw_execute(forwardPlan);

        vector<complex<double>> result(size);
        for (int i = 0; i < size; i++) {
            result[i] = complex<double>(outData[i][0], outData[i][1]);
        }

        return result;
    }

    vector<complex<double>> inverse(vector<complex<double>>& input) {
        if (input.size() != size) {
            cerr << "Ошибка: Размер входных данных должен соответствовать размеру FFT." << endl;
            return vector<complex<double>>();
        }

        for (int i = 0; i < size; i++) {
            inData[i][0] = input[i].real();
            inData[i][1] = input[i].imag();
        }

        fftw_execute(inversePlan);

        vector<complex<double>> result(size);
        for (int i = 0; i < size; i++) {
            result[i] = complex<double>(inData[i][0] / size, inData[i][1] / size);
        }

        return result;
    }

    ~FourierTransform() {
        fftw_destroy_plan(forwardPlan);
        fftw_destroy_plan(inversePlan);
        fftw_free(inData);
        fftw_free(outData);
    }

private:
    int size;
    fftw_complex* inData, * outData;
    fftw_plan forwardPlan, inversePlan;
};

int main() {
    setlocale(LC_ALL, "Russian");
    const int size = 8; // Замените этот размер на нужный (кратный 2, 3 или 5)
    FourierTransform fft(size);

    // Генерируем случайные комплексные входные данные
    vector<complex<double>> input(size);
    for (int i = 0; i < size; i++) {
        input[i] = complex<double>(rand() % 10, rand() % 10);
    }

    // Прямое преобразование Фурье
    vector<complex<double>> fftResult = fft.forward(input);

    // Обратное преобразование Фурье
    vector<complex<double>> ifftResult = fft.inverse(fftResult);

    // Сравнение ошибки между входными и выходными данными
    double error = 0.0;
    for (int i = 0; i < size; i++) {
        error += abs(input[i] - ifftResult[i]);
    }
    error /= size;

    cout << "Средняя ошибка: " << error << endl;

    return 0;
}