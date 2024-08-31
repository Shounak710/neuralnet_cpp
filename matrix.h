#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>
#include <thread>

// Matrix of floating numbers (or any other type T)
template<typename T>
struct Matrix {
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

    // Calculate mean value for each column in matrix
    Matrix<T> col_mean() const {
        Matrix<T> res(1, col_count);

        for(int i=0; i < row_count; i++) {
            for(int j=0; j < col_count; j++) {
                if(i == 0) {
                    res[0][j] = data[i][j] / row_count; // Doing this explicitly due to type problems
                } else {
                    res[0][j] = res[0][j] + (data[i][j] / row_count);
                }
            }
        }

        return res;
    }

    Matrix<T> operator+(const Matrix<T>& other) const {
        if ((other.row_count != row_count) || (other.col_count != col_count)) {
            throw std::invalid_argument(
                "Matrix dimensions are not compatible for addition. \
              Matrix 1: " + this->shape() + "). \
              Matrix 2: " + other.shape());
        }

        Matrix<T> res(row_count, col_count);
        for (size_t i = 0; i < row_count; i++) {
            for (size_t j = 0; j < col_count; j++) {
                res[i][j] = data[i][j] + other[i][j];
            }
        }
        return res;
    }

    Matrix<T> operator+=(const Matrix<T>& other) {
        *this = *this + other;
    }

    Matrix<T> operator-(const Matrix<T>& other) const {
        if ((other.row_count != row_count) || (other.col_count != col_count)) {
            throw std::invalid_argument(
              "Matrix dimensions are not compatible for subtraction. \
              Matrix 1: " + this->shape() + "). \
              Matrix 2: " + other.shape());
        }

        Matrix<T> res(row_count, col_count);
        for (size_t i = 0; i < row_count; i++) {
            for (size_t j = 0; j < col_count; j++) {
                res[i][j] = data[i][j] - other[i][j];
            }
        }
        return res;
    }

    static void matrix_multiply_thread(int start_row, int end_row, const std::vector<std::vector<T>>& s_data,
                           const std::vector<std::vector<T>>& other_data, std::vector<std::vector<T>>& res_data) {
        for (int i = start_row; i < end_row; ++i) {
            for (int j = 0; j < other_data[0].size(); ++j) {
                int sum = 0;
                for (int k = 0; k < s_data[0].size(); ++k) {
                    sum += s_data[i][k] * other_data[k][j];
                }
                res_data[i][j] = sum;
            }
        }
    }

    Matrix<T> operator*(const Matrix<T>& other) const {
        if (col_count != other.row_count) {
            throw std::invalid_argument(
              "Matrix dimensions are not compatible for multiplication. \
              Matrix 1: " + this->shape() + "). \
              Matrix 2: " + other.shape());
        }

        Matrix<T> res(row_count, other.col_count);
        
        int num_threads = std::thread::hardware_concurrency(); // Number of threads supported by the hardware
        std::vector<std::thread> threads;
        int rows_per_thread = row_count / num_threads;

        for (int t = 0; t < num_threads; ++t) {
            int start_row = t * rows_per_thread;
            int end_row = (t == num_threads - 1) ? row_count : start_row + rows_per_thread;
            threads.emplace_back(matrix_multiply_thread, start_row, end_row, std::cref(data), std::cref(other.data), std::ref(res.data));
        }

        for (auto& th : threads) {
            th.join();
        }

        // for (size_t i = 0; i < row_count; i++) {
        //     for (size_t j = 0; j < other.col_count; j++) {
        //         T el;
        //         // std::cout << "starting el: " << el << std::endl;

        //         for (size_t k = 0; k < col_count; k++) {
        //             // std::cout << "i: " << i << " k: " << k << " j: " << j << std::endl;
        //             el = el + data[i][k] * other[k][j];
        //             // std::cout << "data: " << data[i][k] << " other: " << other[k][j] << " el: " << el << std::endl;
        //         }
        //         res[i][j] = el;
        //         el = 0;
        //     }
        // }
        return res;
    }

    Matrix<T> operator*(const std::vector<T>& other) const {
        if (col_count != other.size()) {
            throw std::invalid_argument(
              "Matrix dimensions are not compatible for matrix-vector multiplication. \
              Matrix 1:  " + this->shape() + ". \
              Vector size: " + std::to_string(other.size()));
        }

        Matrix<T> res(row_count, other.size());
        for (size_t i = 0; i < row_count; i++) {
            for (size_t j = 0; j < other.size(); j++) {
                T el = 0;
                for (size_t k = 0; k < col_count; k++) {
                    el += data[i][k] * other[k];
                }
                res[i][j] = el;
            }
        }
        return res;
    }

    Matrix<T> operator/(const T num) const {
        if(num == 0) throw std::runtime_error("Cannot divide matrix by 0");

        Matrix<T> res(row_count, col_count);

        for(size_t i = 0; i < row_count; i++) {
            for(size_t j=0; j < col_count; j++) {
                res[i][j] = data[i][j] / num;
            }
        }

        return res;
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

    static void matrix_el_mult_thread(int start_row, int end_row, const std::vector<std::vector<T>>& s_data,
                           const std::vector<std::vector<T>>& other_data, std::vector<std::vector<T>>& res_data) {
        for(int i=start_row; i < end_row; i++) {
            for(int j=0; j < s_data[0].size(); j++) {
                res_data[i][j] = s_data[i][j] * other_data[i][j];
            }
        }
    }

    Matrix<T> el_mult(Matrix<T>& other) const {
        if ((row_count != other.row_count) || (col_count != other.col_count)) {
            throw std::invalid_argument(
              "Matrix dimensions are not compatible for element-wise multiplication. \
              Matrix 1: " + this->shape() + "). \
              Matrix 2: " + other.shape());
        }

        Matrix<T> res(row_count, col_count);

        int num_threads = std::thread::hardware_concurrency(); // Number of threads supported by the hardware
        std::vector<std::thread> threads;
        int rows_per_thread = row_count / num_threads;

        for (int t = 0; t < num_threads; ++t) {
            int start_row = t * rows_per_thread;
            int end_row = (t == num_threads - 1) ? row_count : start_row + rows_per_thread;
            threads.emplace_back(matrix_el_mult_thread, start_row, end_row, std::cref(data), std::cref(other.data), std::ref(res.data));
        }

        for (auto& th : threads) {
            th.join();
        }

        return res;
    }

    // Multiply matrix with a scalar
    Matrix<T> scalar_mult(float s, bool print=false) const {
        Matrix<T> res(row_count, col_count);

        for(int i=0; i < data.size(); i++) {
            transform(data[i].begin(), data[i].end(), res[i].begin(), [&s](T e) {
                return s * e;
            });
        }

        if(print){
            std::cout << "scalar: " << s << std::endl;
            std::cout << *this << std::endl;

            std::cout << res << std::endl;
        }
        return res;
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
