package a0312;

public class VarEx5 {
    public static void main(String[] args) {
        String item = "라면";
        int price = 800;
        double weight = 0.12;
        boolean discounted = false;

        // System.out.println("상품-" + item + " 가격-" + price + "원 무게-" + weight + "kg 할인여부" + discounted);
        System.out.printf("상품-%s 가격-%d원 무게-%.2f 할인여부-%b",
        item, price, weight, discounted);
    }
    
}
