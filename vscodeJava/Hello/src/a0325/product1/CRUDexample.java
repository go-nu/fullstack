package a0325.product1;

import java.util.Scanner;

public class CRUDexample {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        ProductManager manager = new ProductManager();
        boolean running = true;

        // 새로운 제품 추가
        // manager.addProcuct("Monitor", 300.0);

        while (running) {
            System.out.println("\n📌 제품 관리 시스템");
            System.out.println("1. 제품 추가");
            System.out.println("2. 제품 목록 보기");
            System.out.println("3. 제품 수정");
            System.out.println("4. 제품 삭제");
            System.out.println("5. 종료");
            System.out.print("선택 > ");
            int choice = sc.nextInt();
            sc.nextLine(); // 개행 문자 제거

            switch (choice) {
                case 1:
                    System.out.print("제품명 입력 : ");                    
                    String name = sc.nextLine();
                    System.out.print("가격 입력 : ");
                    double price = sc.nextDouble();
                    manager.addProcuct(name, price);
                    break;
                case 2:
                    manager.listProducts();
                    break;
                case 3:
                    System.out.print("수정할 제품 ID 입력 : ");
                    int idToUpdate = sc.nextInt();
                    sc.nextLine();
                    System.out.print("새로운 제품명 : ");
                    String newName = sc.nextLine();
                    System.out.print("새로운 가격 : ");
                    double newPrice = sc.nextDouble();
                    manager.updateProduct(idToUpdate, newName, newPrice);
                    break;
                case 4:
                    System.out.print("삭제할 제품 ID 입력 : ");
                    int idToDelete = sc.nextInt();
                    sc.nextLine();
                    if (manager.deleteProduct(idToDelete)) {
                        System.out.println("제품이 삭제되었습니다.");
                    } else {
                        System.out.println("해당 제품이 없습니다.");
                    }
                    break;
                case 5:
                    System.out.println("🔚 프로그램을 종료합니다.");
                    running = false;
                    break;
                default:
                    break;
            }
        }
    }
}
