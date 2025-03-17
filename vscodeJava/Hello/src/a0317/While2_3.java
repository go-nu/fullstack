package a0317;

public class While2_3 {
    public static void main(String[] args) {
        int sum = 0;
        int i = 1;

        while(i <= 10) {
            sum += i;
            System.out.println("i = " + i + " sum = " + sum); 
            i++;
        }

    }
}
