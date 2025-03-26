package a0326.coffee1;

import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class CoffeeOrder1 {
    public static void main(String[] args) {
        Map<String, Integer> menu = new HashMap<>();
        
        menu.put("Americano", 3000);
        menu.put("Latte", 4000);
        menu.put("Mocha", 4500);
        menu.put("Espresso", 2500);
        
        Map<String, Integer> order = new HashMap<>();
        Scanner scanner = new Scanner(System.in);

        System.out.print("몇 명의 주문을 받으시겠습니까? ");
        int numOfPeople = scanner.nextInt();
        for(int i = 0; i < numOfPeople; i++) {
            System.out.printf("%d번째 고객님 주문을 시작합니다.", (i+1));

            while (true) {
                System.out.println("\n메뉴 : ");
                for(Map.Entry<String, Integer> entry : menu.entrySet()) {
                    System.out.println(entry.getKey() + " - " + entry.getValue() + "원");
                }
                System.out.print("주문할 커피 이름(종료 : exit) : ");
                String coffee = scanner.nextLine();
                if(coffee.equals("exit")) break;
                if (!menu.containsKey(coffee)) {
                    System.out.println("해당 커피는 메뉴에 없습니다. 다시 입력하세요.");
                    continue;
                }
                System.out.print("수량 : ");
                int quantity;
                while (true) {
                    try {
                        quantity = Integer.parseInt(scanner.nextLine());
                        if (quantity <= 0) {
                            System.out.println("1이상의 숫자를 입력");
                        }
                        break;
                    } catch (NumberFormatException e) {
                        System.out.println("유효한 숫자르 입력해주세요.");
                        continue;
                    }
                }
                
                order.put(coffee, order.getOrDefault(coffee, 0) + quantity);
                System.out.println(coffee + " " + quantity + "개 추가 되었습니다.");
    
            }

        }

        scanner.close();
        System.out.println("\n주문 내역");
        int total = 0;
        for(Map.Entry<String, Integer> entry : order.entrySet()) {
            int price = menu.get(entry.getKey()) * entry.getValue();
            System.out.println(entry.getKey() + " X " + entry.getValue() + " = " + price + "원");
            total += price;
        }
        System.out.println("총 금액 : " + total + "원");
        if(total >= 20000) {
            System.out.println("할인 적용: 10% 할인 - " + total*0.1);
            System.out.println("총 금액: " + total*0.9);
        }

    }
}
