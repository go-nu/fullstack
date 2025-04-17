#include <iostream>
#include <stdexcept>

using namespace std;  // std:: 생략 가능

class Calculator {
public:
    int num1;
    int num2;

    void setNumbers(int n1, int n2) {
        num1 = n1;
        num2 = n2;
    }

    int add() {
        return num1 + num2;
    }

    int subtract() {
        return num1 - num2;
    }

    int multiply() {
        return num1 * num2;
    }

    double divide() {
        if (num2 == 0) {
            throw runtime_error("Error: Division by zero!");  // std:: 없이 사용
        }
        return static_cast<double>(num1) / num2;
    }
};

int main() {
    Calculator calc;
    int n1, n2;
    char operation;

    cout << "두 개의 정수를 입력하세요: ";
    cin >> n1 >> n2;
    calc.setNumbers(n1, n2);

    cout << "수행할 연산을 입력하세요 (+, -, *, /): ";
    cin >> operation;

    try {
        double result;

        switch (operation) {
            case '+':
                result = calc.add();
                break;
            case '-':
                result = calc.subtract();
                break;
            case '*':
                result = calc.multiply();
                break;
            case '/':
                result = calc.divide();  // 예외 발생 가능
                break;
            default:
                throw invalid_argument("Error: Invalid operation!");  // std:: 없이 사용
        }

        cout << "결과: " << result << endl;
    }
    catch (const exception& e) {
        cerr << "예외 발생: " << e.what() << endl;
        return 1;
    }

    return 0;
}
