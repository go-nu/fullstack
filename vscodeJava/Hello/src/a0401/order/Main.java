package a0401.order;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

class Customer {
    private final String name;
    private final String city;

    public Customer(String name, String city) {
        this.name = name;
        this.city = city;
    }

    public String getName() {
        return name;
    }

    public String getCity() {
        return city;
    }

    @Override
    public String toString() {
        return "Customer{" +
                "name='" + name + '\'' +
                ", city='" + city + '\'' +
                '}';
    }
}

class Order {
    private final int id;
    private final Customer customer;
    private final String product;
    private final int quantity;

    public Order(int id, Customer customer, String product, int quantity) {
        this.id = id;
        this.customer = customer;
        this.product = product;
        this.quantity = quantity;
    }

    public int getId() {
        return id;
    }

    public Customer getCustomer() {
        return customer;
    }

    public String getProduct() {
        return product;
    }

    public int getQuantity() {
        return quantity;
    }

    @Override
    public String toString() {
        return "Order{" +
                "id=" + id +
                ", customer=" + customer +
                ", product='" + product + '\'' +
                ", quantity=" + quantity +
                '}';
    }
}

public class Main {
    public static void main(String[] args) {
        Customer customer1 = new Customer("Kim", "Seoul");
        Customer customer2 = new Customer("Lee", "Busan");
        Customer customer3 = new Customer("Park", "Seoul");
        Customer customer4 = new Customer("Choi", "Seoul");

        List<Order> orders = Arrays.asList(
                new Order(1, customer1, "Laptop", 1),
                new Order(2, customer2, "Smartphone", 2),
                new Order(3, customer3, "Keyboard", 1),
                new Order(4, customer1, "Mouse", 3),
                new Order(5, customer4, "Monitor", 1),
                new Order(6, customer3, "USB Cable", 2)
        );

        // 여기에 답을 작성하세요.
        // 1. 서울에 사는 고객의 주문 출력
        System.out.println("1 >>");
        answer1(orders);
        System.out.println();
        // 2. 모든 주문의 총 수량 구하기(mapToInt)
        System.out.println("2 >>");
        answer2(orders);
        System.out.println();
        // 3. 각 고객의 이름과 그 고객이 주문한 제품명 출력
        System.out.println("3 >>");
        answer3(orders);
        System.out.println();
        // 4. 특정 고객의 주문만 필터링하여 출력
        System.out.println("4 >>");
        answer4(orders, "Kim");
        System.out.println();
        // 5. 주문 수량이 2개 이상인 주문만 출력
        System.out.println("5 >>");
        answer5(orders);
        System.out.println();
        // 6. 고객이 주문한 제품의 종류를 중복 없이 출력
        System.out.println("6 >>");
        answer6(orders);
        System.out.println();
        // 7. 모든 주문을 수량 기준 내림차순으로 정렬
        System.out.println("7 >>");
        answer7(orders);
        System.out.println();
        // 8. 각 도시별 고객 수 출력
        System.out.println("8 >>");
        answer8(orders);
        System.out.println();
    }

    private static void answer1(List<Order> orders) {
        List<Order> result = orders.stream()
            .filter(o -> "Seoul".equals(o.getCustomer().getCity()))
            .collect(Collectors.toList());
        System.out.println(result);
    }

    private static void answer2(List<Order> orders) {
        int sum1 = orders.stream()
            .mapToInt(Order::getQuantity)
            .sum();
        System.out.println(sum1);
    }

    private static void answer3(List<Order> orders) {
        List<String> names = orders.stream()
            .map(Order::getCustomer)
            .map(Customer::getName)
            .distinct()
            .collect(Collectors.toList());
        for (String n : names) {
            System.out.print(n + ", ");
            List<String> os = orders.stream()
                .filter(o -> n.equals(o.getCustomer().getName()))
                .map(Order::getProduct)
                .collect(Collectors.toList());
            System.out.println(os);
        }
    }

    private static void answer4(List<Order> orders, String n) {
        List<Order> result = orders.stream()
            .filter(o -> n.equals(o.getCustomer().getName()))
            .collect(Collectors.toList());
        System.out.println(result);
    }

    private static void answer5(List<Order> orders) {
        List<Order> result = orders.stream()
            .filter(o -> (o.getQuantity() > 2))
            .collect(Collectors.toList());
        System.out.println(result);
    }

    private static void answer6(List<Order> orders) {
        List<String> result = orders.stream()
            .map(Order::getProduct)
            .distinct()
            .collect(Collectors.toList());
        System.out.println(result);
    }

    private static void answer7(List<Order> orders) {
        List<Order> result = orders.stream()
            .sorted(Comparator.comparing(Order::getQuantity).reversed())
            .collect(Collectors.toList());
        System.out.println(result);
    }

    private static void answer8(List<Order> orders) {
        List<String> citys = orders.stream()
            .map(Order::getCustomer)
            .map(Customer::getCity)
            .distinct()
            .collect(Collectors.toList());
        for (String c : citys) {
            System.out.print(c + ", ");
            int count = (int)orders.stream()
                .map(Order::getCustomer)
                .distinct()
                .filter(o -> c.equals(o.getCity())).count();
            System.out.println(count);
        }
    }

}