#include <stdio.h>

struct Student {
    char name[20];
    int studentId;
    char grade;
};

int main() {
    struct Student s;
    printf("이름을 입력하세요(최대 20자) : ");
    scanf("%s", s.name);
    printf("학번을 입력하세요 : ");
    scanf("%d", &s.studentId);
    printf("학점을 입력하세요 : ");
    scanf(" %c", &s.grade); // 맨 앞 공백 문자로 scanf 오류 방지

    printf("\n-----학생 정보-----\n");
    printf("이름: %s\n", s.name);
    printf("학번: %d\n", s.studentId);
    printf("학점: %c\n", s.grade);
    
    return 0;
}