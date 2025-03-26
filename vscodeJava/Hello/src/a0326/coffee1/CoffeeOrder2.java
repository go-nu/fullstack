package a0326.coffee1;

import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class CoffeeOrder2 {
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
        scanner.nextLine();
        for(int i = 0; i < numOfPeople; i++) {
            System.out.printf("%d번째 고객님 주문을 시작합니다.\n", (i+1));

            while(true) {
                System.out.println("메뉴: ");
                for(Map.Entry<String, Integer> m : menu.entrySet()) {
                    System.out.printf("%s - %d원\n", m.getKey(), m.getValue());
                }
                System.out.print("주문할 커피 이름(종료: exit): ");
                String oMenu = scanner.nextLine();
                scanner.nextLine();
                if (oMenu.equals("exit")) {
                    break;
                }
                System.out.print("수량: ");
                int oCount = scanner.nextInt();
                scanner.nextLine();
                System.out.printf("%s가 %d개 추가되었습니다.\n", oMenu, oCount);
                order.put(oMenu, order.getOrDefault(oMenu, 0) + oCount);
            }
        }
        scanner.close();
        System.out.println("주문 내역: ");
        int total = 0;
        for(Map.Entry<String, Integer> o : order.entrySet()) {
            System.out.println(o.getKey() + " x " + o.getValue() + " = " + menu.get(o.getKey())*o.getValue());
            total += menu.get(o.getKey())*o.getValue();
        }
        System.out.println("총 금액: " + total);
        if(total >= 20000) {
            System.out.println("할인 적용: 10% 할인 - " + total*0.1);
            System.out.println("총 금액: " + total*0.9);
        }
    }
}
