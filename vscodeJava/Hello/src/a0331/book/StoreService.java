package a0331.book;

import java.text.DecimalFormat;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Scanner;

public class StoreService {
    Book book = Book.getInstance();
    Customer customer;
    Scanner s = new Scanner(System.in);
    boolean reOrder = false;
    Map<String, Integer> orderList;
    
    public StoreService() {
        orderList = new LinkedHashMap<>();
    }

    int orderNum = 1;

    public void start() {
        System.out.println("어서오세요");
        customer = new Customer(orderNum);
        book.getMenu();
        order();
        totalOrder(customer);
    }

    private void order() {
        end:while (true) {
            try {
                System.out.println("\n원하는 도서의 번호를 입력해주세요");
                System.out.print("취소를 원하시면 0번을 눌러주세요 : ");
                int select = s.nextInt();
                s.nextLine();
                if(select == 0) {
                    System.out.println("주문을 취소합니다.");
                    System.exit(0); // 종료
                }
                String bBookName = book.bookList.get(select-1);
                System.out.println("선택하신 책은 " + bBookName + "입니다. 몇 권 구매하시겠습니까?");
                int bBookCount = s.nextInt();
                s.nextLine();

                if(reOrder) {
                    for(String oBookName : orderList.keySet()) {
                        if (oBookName.equalsIgnoreCase(bBookName)) {
                            int newCount = orderList.get(oBookName).intValue() + bBookCount;
                            orderList.replace(bBookName, newCount);
                        } else {
                            orderList.put(bBookName, bBookCount);
                            break;
                        }
                    }
                } else {
                    orderList.put(bBookName, bBookCount);
                }
                addOrder(); // 추가 주문 확인
                customer.setBookOrder(orderList); // 책 주문 객체 초기화
                break end; // 종료

            } catch (Exception e) {
                System.out.println("잘못된 입력입니다.");
                e.printStackTrace();
                continue; // while문 복귀
            }
        }
    }

    private void addOrder() {
        System.out.println("추가 주문을 하시겠습니까?");
        System.out.print("예(Y) / 아니오 (N) : ");
        String answer = s.nextLine();
        if(answer.equals("예") || answer.equalsIgnoreCase("y")) {
            book.getMenu();
            reOrder = true;
            order();
        } else if (answer.equals("아니오") || answer.equalsIgnoreCase("n")) {
            return; // addOrder를 불러온 order로 복귀
        }
    }

    private void totalOrder(Customer customer) {
        int orderIndex = 1;
        int totalMoney = 0;
        int bookPrice = 0;
        DecimalFormat f = new DecimalFormat("###,000원");
        String cName = customer.getOrderName();
        StringBuffer sb = new StringBuffer();
        sb.append("\n\n")
        .append("+----------------------------------------------------+\n ")
        .append(cName + "님의 주문 내역 입니다" + "\n");

        for(Map.Entry<String, Integer> order : customer.getBookOrder().entrySet()) {
            bookPrice = book.menu.get(order.getKey()) * order.getValue();
            sb.append(String.format("[%d] %s : %d권, %d원\n", orderIndex, order.getKey(), order.getValue(), bookPrice));
            orderIndex++;
            totalMoney += bookPrice;
        }
        sb.append("+----------------------------------------------------+\n ")
        .append("총 결제 금액은 " + f.format(totalMoney) + "원 입니다.");
        System.out.println(sb);
        payment(totalMoney);
    }

    private void payment(int totalMoney) {
        System.out.println("결제 도와드리겠습니다.");
        int payResult = customer.getMoney() - totalMoney;
        if(payResult < 0) {
            System.out.println("잔액이 부족합니다. 확인 후 다시 시도해주세요.");
        } else {
            customer.setMoney(payResult);
            System.out.println("결제가 완료되었습니다.\n 이용해주셔서 감사합니다.");
            orderNum++;
        }
    }
}
