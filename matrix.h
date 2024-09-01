#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>
#include <thread>
#include <cmath>

// Matrix of floating numbers (or any other type T)
template<typename T>
struct Matrix {
    private:
        static void matrix_add_thread(int start_col, int end_col, const std::vector<std::vector<T>>& s_data,
                            const std::vector<std::vector<T>>& other_data, std::vector<std::vector<T>>& res_data) {
            for (int i = 0; i < s_data.size(); i++) {
                for (int j = start_col; j < end_col; ++j) {
                    res_data[i][j] = s_data[i][j] + other_data[i][j];
                }
            }
        }

        static void matrix_subtract_thread(int start_col, int end_col, const std::vector<std::vector<T>>& s_data,
                           const std::vector<std::vector<T>>& other_data, std::vector<std::vector<T>>& res_data) {
            for (int i = 0; i < s_data.size(); i++) {
                for (int j = start_col; j < end_col; ++j) {
                    res_data[i][j] = s_data[i][j] - other_data[i][j];
                }
            }
        }

        static void matrix_multiply_thread(int start_row, int end_row, const std::vector<std::vector<T>>& s_data,
                            const std::vector<std::vector<T>>& other_data, std::vector<std::vector<T>>& res_data) {
            for (int i = start_row; i < end_row; ++i) {
                for (int j = 0; j < other_data[0].size(); ++j) {
                    for (int k = 0; k < s_data[0].size(); ++k) {
                        res_data[i][j] += s_data[i][k] * other_data[k][j];
                    }
                }
            }
        }

        static void matrix_el_mult_thread(int start_row, int end_row, const std::vector<std::vector<T>>& s_data,
                           const std::vector<std::vector<T>>& other_data, std::vector<std::vector<T>>& res_data) {
            for(int i=start_row; i < end_row; i++) {
                for(int j=0; j < s_data[0].size(); j++) {
                    res_data[i][j] = s_data[i][j] * other_data[i][j];
                }
            }
        }

        // passing other data to this function even if it's not needed to keep same parameters in all threading functions
        // this allows creating an abstract 'threadify' function which is cleaner for the code.
        static void matrix_col_mean_thread(int start_col, int end_col, const std::vector<std::vector<T>>& s_data,
                           const std::vector<std::vector<T>>& other_data, std::vector<std::vector<T>>& res_data) {
            for(int i=0; i < s_data.size(); i++) {
                for(int j=start_col; j < end_col; j++) {
                    if(i == 0) {
                        res_data[0][j] = s_data[i][j] / s_data.size();
                    } else {
                        res_data[0][j] += (s_data[i][j] / s_data.size());
                    }
                }
            }
        }

        // passing other data to this function even if it's not needed to keep same parameters in all threading functions
        // this allows creating an abstract 'threadify' function which is cleaner for the code.
        static void matrix_pow_thread(int start_row, int end_row, const std::vector<std::vector<T>>& s_data,
                           const std::vector<std::vector<T>>& other_data, std::vector<std::vector<T>>& res_data) {
            for(int i=start_row; i < end_row; i++) {
                for(int j=0; j < s_data[0].size(); j++) {
                    res_data[i][j] = std::pow(s_data[i][j], other_data[0][0]);
                }
            }
        }

        static void threadify(void (*thread_func)(int, int, const std::vector<std::vector<T>>&, const std::vector<std::vector<T>>&,
        std::vector<std::vector<T>>&), const size_t total_threads, const std::vector<std::vector<T>>& s_data,
                     const std::vector<std::vector<T>>& other_data, std::vector<std::vector<T>>& res_data) {

            int num_threads = std::thread::hardware_concurrency(); // Number of threads supported by the hardware

            std::vector<std::thread> threads;

            int datapts_per_thread = total_threads / num_threads;

            for (int t = 0; t < num_threads; ++t) {
                int start_idx = t * datapts_per_thread;
                int end_idx = (t == num_threads - 1) ? total_threads : start_idx + datapts_per_thread;
                threads.emplace_back(thread_func, start_idx, end_idx, std::cref(s_data), std::cref(other_data), std::ref(res_data));
            }

            for (auto& th : threads) {
                th.join();
            }
        }

    public:
    std::vector<std::vector<T>> data;
    size_t row_count = 0, col_count = 0;

    Matrix(): row_count(0), col_count(0) {}

    Matrix(int rc, int cc) : row_count(rc), col_count(cc), data(rc, std::vector<T>(cc)) {}
    Matrix(int rc, int cc, T default_val): row_count(rc), col_count(cc), data(rc, std::vector<T>(cc, default_val)) {} 

    Matrix(const std::vector<std::vector<T>>& input_data) : data(input_data) {
        row_count = data.size();
        if (row_count > 0) col_count = data[0].size();
    }

    void update_shape() {
        // After directly pushing into data vector, matrix shape should be updated too
        row_count = data.size();

        if(data.size() > 0) {
            col_count = data[0].size();
        } else {
            col_count = 0;
        }
    }

    std::vector<T>& operator[](size_t row) {
        return data[row];
    }

    const std::vector<T>& operator[](size_t row) const {
        return data[row];
    }

    Matrix<T> transpose() const {
        Matrix<T> res(col_count, row_count);
        for (size_t i = 0; i < row_count; i++) {
            for (size_t j = 0; j < col_count; j++) {
                res[j][i] = data[i][j];
            }
        }
        return res;
    }

    Matrix<T> Tr() const { return transpose(); }

    Matrix<T> operator+(const Matrix<T>& other) const {
        if ((other.row_count != row_count) || (other.col_count != col_count)) {
            throw std::invalid_argument(
                "Matrix dimensions are not compatible for addition. \
              Matrix 1: " + this->shape() + "). \
              Matrix 2: " + other.shape());
        }

        Matrix<T> res(row_count, col_count);
        threadify(matrix_add_thread, col_count, std::cref(data), std::cref(other.data), std::ref(res.data));

        return res;
    }

    Matrix<T> operator+=(const Matrix<T>& other) {
        *this = *this + other;
        return *this;
    }

    Matrix<T> operator-(const Matrix<T>& other) const {
        if ((other.row_count != row_count) || (other.col_count != col_count)) {
            throw std::invalid_argument(
              "Matrix dimensions are not compatible for subtraction. \
              Matrix 1: " + this->shape() + "). \
              Matrix 2: " + other.shape());
        }

        Matrix<T> res(row_count, col_count);
        threadify(matrix_subtract_thread, col_count, std::cref(data), std::cref(other.data), std::ref(res.data));

        return res;
    }

    Matrix<T> operator*(const Matrix<T>& other) const {
        if (col_count != other.row_count) {
            throw std::invalid_argument(
              "Matrix dimensions are not compatible for multiplication. \
              Matrix 1: " + this->shape() + "). \
              Matrix 2: " + other.shape());
        }

        Matrix<T> res(row_count, other.col_count);
        threadify(matrix_multiply_thread, row_count, std::cref(data), std::cref(other.data), std::ref(res.data));

        return res;
    }

    Matrix<T> operator/(const T num) const {
        if(num == 0) throw std::runtime_error("Cannot divide matrix by 0");

        return this->scalar_mult(1/num);
    }

    Matrix<T>& operator=(const Matrix<T>& other) {
        row_count = other.row_count;
        col_count = other.col_count;

        data = other.data;

        return *this;
    }

    Matrix<T>& operator=(const T& val) {
        for(auto i : data) {
            for(auto j : i) {
                j = val;
            }
        }

        return *this;
    }

    Matrix<T> el_mult(Matrix<T>& other) const {
        if ((row_count != other.row_count) || (col_count != other.col_count)) {
            throw std::invalid_argument(
              "Matrix dimensions are not compatible for element-wise multiplication. \
              Matrix 1: " + this->shape() + "). \
              Matrix 2: " + other.shape());
        }

        Matrix<T> res(row_count, col_count);
        threadify(matrix_el_mult_thread, row_count, std::cref(data), std::cref(other.data), std::ref(res.data));

        return res;
    }

    Matrix<T> pow(T exp) const {
        Matrix<T> res(row_count, col_count);
        std::vector<std::vector<T>> other_data = {{exp}};
        threadify(matrix_pow_thread, row_count, std::cref(data), std::cref(other_data), std::ref(res.data));

        return res;
    }
    // Multiply matrix with a scalar
    Matrix<T> scalar_mult(T s, bool print=false) const {
        Matrix<T> other_data(row_count, col_count, s); 
        return this->el_mult(other_data);
    }

    // Treat each row of self and other matrix as a separate matrix, and multiply after aligning correctly by transposing as needed.
    Matrix<Matrix<T>> row_mult(const Matrix<T>& other, bool transpose_self=true, bool transpose_other=false) const {
        if ((row_count != other.row_count)) {
            throw std::invalid_argument(
              "Matrix dimensions are not compatible for row multiplication. \
              Matrix 1: " + this->shape() + " \
              Matrix 2: " + other.shape());
        }

        Matrix<Matrix<T>> res(row_count, 1);

        for(int i=0; i < row_count; i++) {
            Matrix<T> mat_1 = Matrix<T>({data[i]});
            Matrix<T> mat_2 = Matrix<T>({other.data[i]});

            if(transpose_self) mat_1 = mat_1.Tr();
            if(transpose_other) mat_2 = mat_2.Tr();

            // std::cout << "i: " << i << " mat 1: " << mat_1.shape() << std::endl << mat_1 << std::endl;
            // std::cout << "i: " << i << " mat 2: " << mat_2.shape() << std::endl << mat_2 << std::endl;
            res[i] = {mat_1 * mat_2};
            // std::cout << "i: " << i << " res: " << std::endl << res[i][0] << std::endl;
        }

        return res;
    }

    float mean() const {
        float sum = 0.0;

        for(int i=0; i < row_count; i++) {
            sum += accumulate(data[i].begin(), data[i].end(), 0.0);
        }

        return sum / (row_count * col_count);
    }

    // Calculate mean value for each column in matrix
    Matrix<T> col_mean() const {
        Matrix<T> res(1, col_count);
        threadify(matrix_col_mean_thread, col_count, std::cref(data), std::cref(data), std::ref(res.data));

        return res;
    }

    std::string shape() const {
        return "(" + std::to_string(row_count) + ", " + std::to_string(col_count) + ")";
    }

    void print() const {
        std::cout << "Printing matrix:" << std::endl;
        for (size_t i = 0; i < row_count; i++) {
            for (size_t j = 0; j < col_count; j++) {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix) {
        for (size_t i = 0; i < matrix.row_count; i++) {
            for (size_t j = 0; j < matrix.col_count; j++) {
                os << matrix[i][j] << " ";
            }
            os << std::endl;
        }
        return os;
    }
};
