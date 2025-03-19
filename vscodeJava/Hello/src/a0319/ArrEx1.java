package a0319;

public class ArrEx1 {
    public static void main(String[] args) {
        String[] city = {"서울", "부산", "인천", "대전", "대구"};
        int[] count = {599, 51, 46, 43, 27};
        for(int i = 0; i <city.length; i++) {
            System.out.printf("%s: %d명\n", city[i],count[i]);
        }
    }
}
