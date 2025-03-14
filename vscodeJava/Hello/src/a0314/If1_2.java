package a0314;

public class If1_2 {
    public static void main(String[] args) {
        int age = 18;
        String status;

        if(age >= 20) 
            status = "성인";
        else 
            status = "미성년자";

        System.out.println("age = " + age + " status = " + status);
    }
}
