#include <stdio.h>

// Function declaration
int sum(int k);

// The main method
int main() {
    int result = sum(10); // call the function
    printf("Result is = %d", result);
    return 0;
}

// Function definition
int sum(int k) {
    if(k > 0) return k + sum(k-1);
    else return 0;
}