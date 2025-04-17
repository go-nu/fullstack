#include <iostream>
using namespace std;
class Calculator{
public:    
    int num1;
    int num2;

    void setNumbers(int n1, int n2){
        num1 = n1;
        num2 = n2;
    }
    int add(){
        return num1 + num2;
    }

    int subtract() {
        return num1 - num2;
    }

    int multiply() {
        return num1 * num2;
    }
    double divide(){
        if(num2 == 0){
            cerr << "Error: division by zero!" << endl;
            return 0.0;
        }

        // return (double)(num1) / num2; c언어 문법
        return static_cast<double>(num1) / num2;
    }

};
int main(){
    Calculator calc;
    int n1, n2; 
    char operation;
    cout << "두개의 정수를 입력하세요: ";
    cin >> n1 >> n2;
    calc.setNumbers(n1, n2);

    cout <<"수행할 연산을 입력하세요(+,-,* ,/):";
    cin >> operation;

    double result;

    switch (operation)
    {
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
        result = calc.divide();
        break;
    default:
        cerr << "Error: Invalid operation!" << endl;
    }
    //cerr c++ 에서 표준 에러 줄력시스템 
    cout << "결과 : " << result << endl;
    return 0;
}

