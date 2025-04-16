#include <iostream>
using namespace std;

#define SEATS 10

void displaySeats(bool seats[]) {
    cout << "현재 좌석 상태:\n";
    for (int i = 0; i < SEATS; i++) {
        cout << "[" << (i + 1) << "] ";
        if (seats[i]) cout << "예약됨 ";
        else cout << "비어있음 ";
    }
    cout << endl << endl;
}

int main() {
    bool seats[SEATS] = { false };  // false = 비어있음, true = 예약됨
    int choice;
    
    while (true) {
        displaySeats(seats);
        cout << "좌석 예약 프로그램\n";
        cout << "1. 좌석 예약\n";
        cout << "2. 좌석 취소\n";
        cout << "3. 프로그램 종료\n";
        cout << "메뉴 선택: ";
        cin >> choice;

        if (choice == 1) {
            int seatNum;
            cout << "예약할 좌석 번호 (1~10): ";
            cin >> seatNum;

            if (seatNum < 1 || seatNum > SEATS) {
                cout << "잘못된 좌석 번호입니다.\n\n";
            } else if (seats[seatNum - 1]) {
                cout << "이미 예약된 좌석입니다.\n\n";
            } else {
                seats[seatNum - 1] = true;
                cout << "좌석 예약이 완료되었습니다!\n\n";
            }

        } else if (choice == 2) {
            int seatNum;
            cout << "취소할 좌석 번호 (1~10): ";
            cin >> seatNum;

            if (seatNum < 1 || seatNum > SEATS) {
                cout << "잘못된 좌석 번호입니다.\n\n";
            } else if (!seats[seatNum - 1]) {
                cout << "이미 비어있는 좌석입니다.\n\n";
            } else {
                seats[seatNum - 1] = false;
                cout << "좌석 예약이 취소되었습니다.\n\n";
            }

        } else if (choice == 3) {
            cout << "프로그램을 종료합니다.\n";
            break;

        } else {
            cout << "잘못된 선택입니다. 다시 입력하세요.\n\n";
        }
    }

    return 0;
}