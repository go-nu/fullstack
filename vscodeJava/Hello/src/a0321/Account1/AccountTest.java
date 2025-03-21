package a0321.Account1;

public class AccountTest {
    public static void main(String[] args) {
        Account acc = new Account();
        // acc.balance = 1000;
        // System.out.printf("잔액: %d", acc.balance);
        acc.setBalance(1000);
        System.out.printf("잔액: %d", acc.getBalance());
    }
}
class Account {
    // 자신 이외 모든 클래스의 접근 거부
    private int balance;
/*
    // private은 외부에서 접근이 불가능하므로 내부에서 method를 통해 접근

    // getter Method
    public int getBalance() {
        return balance;
    }
    // setter Method
    public void setBalance(int b) {
        balance = b;
    }
 */

    public int getBalance() {
        return balance;
    }

    public void setBalance(int balance) {
        this.balance = balance;
    }
}
