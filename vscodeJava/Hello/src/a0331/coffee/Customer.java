package a0331.coffee;

import java.util.Map;

public class Customer {
    private int orderNum; // 주문 번호
    private int money; // 고객이 보유한 돈
    private Map<String, Integer> coffeeOrder;
    // 고객이 주문한 커피의 종류와 수량을 저장
    // <아메리카노, 2>, <아이스 아메리카노, 3>

    public Customer(int orderNum) {
        this.orderNum = orderNum;
        this.money = 20000; // 고객이 가지고 있는 돈, 초기 20000원
    }

    public Map<String, Integer> getCoffeeOrder() {
        return coffeeOrder;
    }

    public void setCoffeeOrder(Map<String, Integer> coffeeOrder) {
        this.coffeeOrder = coffeeOrder;
    }

    public int getOrderNum() {
        return orderNum;
    }

    public void setOrderNum(int orderNum) {
        this.orderNum = orderNum;
    }

    public int getMoney() {
        return money;
    }

    public void setMoney(int money) {
        this.money = money;
    }

    public String getOrderName() { // 고객1 고객2 고객3
        return orderNum + " ";
    }


    

}
