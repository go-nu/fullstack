package a0321.Bank;

import java.util.Scanner;

public class BankApplication {
    private static Account[] accountArray = new Account[100];
    // 모든 계좌정보를 저장하는 array -> 하나의 배열에 저장하므로 새로운 배열을 또 만들 필요가 없어 static을 사용.
    private static Scanner scanner = new Scanner(System.in);
    
    public static void main(String[] args) {
        boolean run = true;
        
        while (run) {
            System.out.println("--------------------------------------------------------");
            System.out.println("1.계좌생성 | 2.계좌목록 | 3.예금 | 4.출금 | 5.이체 | 6.종료");
            System.out.println("--------------------------------------------------------");
            System.out.print("선택>");
            int selectNo = Integer.parseInt(scanner.nextLine());
            if(selectNo == 1) {
                createAccount();
            } else if (selectNo == 2) {
                accountList();
            } else if (selectNo == 3) {
                deposit();
            } else if (selectNo == 4) {
                withdraw();
            } else if (selectNo == 5) {
                transfer();
            } else if (selectNo == 6) {
                run = false;
            }
        }
        System.out.println("프로그램 종료");
    }

    private static void createAccount() {
        System.out.println("--------");
        System.out.println("계좌생성");
        System.out.println("--------");
        System.out.print("계좌번호: ");
        String ano = scanner.nextLine();
        System.out.print("계좌주: ");
        String oner = scanner.nextLine();
        System.out.print("초기입금액: ");
        int balance = Integer.parseInt(scanner.nextLine());
        
        Account newAccount = new Account(ano, oner, balance); // 키보드로 입력된 값을 생성자로 초기화
        for(int i = 0; i < accountArray.length; i++) {
            if(accountArray[i] == null) { // 비어있는 배열이면 
                accountArray[i] = newAccount;
                System.out.println("결과 : 계좌가 생성되었습니다.");
                break;
            }
        }
        // accountArray[0] = newAccount("110-110", "gildong", 10000);
        // accountArray[0] = newAccount("110-111", "sunja", 20000);
    }

    private static void accountList() {
        System.out.println("--------");
        System.out.println("계좌목록");
        System.out.println("--------");
        for(int i = 0; i < accountArray.length; i++) {
            Account account = accountArray[i];
            if(account != null) {
                System.out.printf("%6s %6s %6d\n", account.getAno(), account.getOner(), account.getBalance());
            }
        }
    }

    private static void deposit() {
        System.out.println("--------");
        System.out.println("예금");
        System.out.println("--------");
        System.out.print("계좌번호: ");
        String ano = scanner.nextLine();
        System.out.print("예금액: ");
        int money = Integer.parseInt(scanner.nextLine());
        Account account = findAccount(ano);
        if(account == null) {
            System.out.println("결과 : 계좌가 없습니다.");
            return;
        } else {
            account.setBalance(account.getBalance() + money);
            // 현재 잔액을 getter로 불러와, 예금액(money)를 더하고 setter로 새로 초기화
        }

    }

    private static void withdraw() {
        System.out.println("--------");
        System.out.println("출금");
        System.out.println("--------");
        System.out.print("계좌번호: ");
        String ano = scanner.nextLine();
        System.out.print("출금액: ");
        int money = Integer.parseInt(scanner.nextLine());
        Account account = findAccount(ano);
        if(account == null) {
            System.out.println("결과 : 계좌가 없습니다.");
            return;
        } else {
            account.setBalance(account.getBalance() - money);
        }

    }

    private static Account findAccount(String ano) {
        Account account = null;
        for(int i = 0; i < accountArray.length; i++) {
            if(accountArray[i] != null) { // 배열이 비어있지 않고,
                String dbAno = accountArray[i].getAno(); // 배열의 계좌번호를 dbAno에 초기화
                if(dbAno.equals(ano)) { // i번째 ano(dbAno)와 입력된 ano가 같으면
                    account = accountArray[i];
                    break;
                }
            }
        }
        return account;
    }

    private static void transfer() {
        System.out.println("--------");
        System.out.println("이체");
        System.out.println("--------");
        System.out.print("입금 계좌번호: ");
        String anoIn = scanner.nextLine();
        System.out.print("출금 계좌번호: ");
        String anoOut = scanner.nextLine();
        System.out.print("출금액: ");
        int money = Integer.parseInt(scanner.nextLine());
        Account accountIn = findAccount(anoIn);
        Account accountOut = findAccount(anoOut);
        if((accountIn != null) && (accountOut != null)) {
            if(accountOut.getBalance() < money) {
                accountIn.setBalance(accountIn.getBalance() + money);
                accountOut.setBalance(accountOut.getBalance() - money);
                return;
            }
            else System.out.println("출금계좌 잔액부족.");
        } else {
            System.out.println("결과 : 계좌가 없습니다.");
        }

    }

}
