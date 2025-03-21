package a0321.Bank;

public class Account {
    private String ano; // 계좌번호
    private String oner; // 계좌주
    private int balance; // 잔액

    public Account(String ano, String oner, int balance) {
        this.ano = ano;
        this.oner = oner;
        this.balance = balance;
    }
    
    public String getAno() {
        return ano;
    }
    public void setAno(String ano) {
        this.ano = ano;
    }
    public String getOner() {
        return oner;
    }
    public void setOner(String oner) {
        this.oner = oner;
    }
    public int getBalance() {
        return balance;
    }
    public void setBalance(int balance) {
        this.balance = balance;
    }
}
