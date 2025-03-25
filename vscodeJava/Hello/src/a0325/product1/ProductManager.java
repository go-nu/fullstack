package a0325.product1;

import java.util.ArrayList;

public class ProductManager {
    private ArrayList<Product> products = new ArrayList<>();
    // Product 객체들을 관리할 ArrayList
    private int nextId = 1; // 제품 id 자동 증가

    // 더미데이터 추가
    public ProductManager(){
        products.add(new Product(nextId++, "Laptop", 1200.0));
        products.add(new Product(nextId++, "Mouse", 50.0));
        products.add(new Product(nextId++, "Keyboard", 100.0));
    }

	public void addProcuct(String name, double price) {
		products.add(new Product(nextId++, name, price));
        System.out.println("제품이 추가되었습니다.");
	}

    public void listProducts() {
        if(products.isEmpty()) { // 비어있으면
            System.out.println("제품이 없습니다.");
        } else {
            for(Product p : products) {
                System.out.println(p.toString());
            }
        }
    }

    public boolean updateProduct(int id, String newName, double newPrice) {
        for(Product p : products) {
            if(p.getId() == id) {
                p.setName(newName);
                p.setPrice(newPrice);
                System.out.println("제품 정보가 수정되었습니다.");
                return true;
            }
        }
        System.out.println("해당 ID의 제품이 없습니다.");
        return false;
    }

    public boolean deleteProduct(int idToDelete) {
        for(Product p : products) {
            if(p.getId() == idToDelete) {
                products.remove(p);
                return true;
            }
        }
        return false;
        // return products.removeIf(p -> p.getId() == id);
    }
}
