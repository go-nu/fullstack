package a0326.coffee1;

import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class CoffeeOrder {
    public static void main(String[] args) {
        Map<String, Integer> menu = new HashMap<>();
        
        menu.put("Americano", 3000);
        menu.put("Latte", 4000);
        menu.put("Mocha", 4500);
        menu.put("Espresso", 2500);
        
        Map<String, Integer> order = new HashMap<>();
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.println("\n메뉴 : ");
            // for(String coffee : menu.keySet()) {
            //     System.out.println(coffee + " - " + menu.get(coffee) + "원");
            // }
            // menu.keySet() - menu의 모든 키
            // menu.get(coffee) -> 키에 해당하는 값을 불러옴

            for(Map.Entry<String, Integer> entry : menu.entrySet()) {
                System.out.println(entry.getKey() + " - " + entry.getValue() + "원");
            }
            // menu.entrySet() - 커피이름과 가격을 저장
            // entry.getKey() - 커피이름, entry.getValue() - 커피가격

            System.out.print("주문할 커피 이름(종료 : exit) : ");
            String coffee = scanner.nextLine();
            if(coffee.equals("exit")) break;
            if (!menu.containsKey(coffee)) { // 입력한 커피이름이 menu의 key에 들어있지 않으면
                System.out.println("해당 커피는 메뉴에 없습니다. 다시 입력하세요.");
                continue;
            }
            System.out.print("수량 : ");
            int quantity;
            while (true) {
                try {
                    quantity = Integer.parseInt(scanner.nextLine());
                    if (quantity >= 0) {
                        System.out.println("1이상의 숫자를 입력");
                    }
                    break;
                } catch (NumberFormatException e) {
                    System.out.println("유효한 숫자르 입력해주세요.");
                    continue;
                }
            }
            

            // order.put(coffee, quantity);
            order.put(coffee, order.getOrDefault(coffee, 0) + quantity); // coffee(key)가 들어오지 않으면 0, 들어 있으면 + quantity
            // getOrDefault()는 Map에서 키가 존재하지 않은 경우, 기본값을 반환하는 매서드
            // null 값을 방지하고, 기본값을 처리
            // if(menu.containsKey(coffee)) {
            //     order.put(coffee, quantity);
            // }
            // containsKey()를 사용할 필요없이 코드가 간결해짐, 위 if문과 같은 동작 수행
            System.out.println(coffee + " " + quantity + "개 추가 되었습니다.");

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

    }
}
