package a0331.sort.hak1;

public class Compare1 {
    public static void main(String[] args) {
        String str1 = "apple";
        String str2 = "banana";
        String str3 = "apple";

        // str1.compareTo(str2) : str1 첫글자의 ASC - str2 첫글자의 ASC
        // 음수 -> 1>2>3 오름차순, 양수 -> 1>2>3 내림차순
        System.out.println(str1.compareTo(str2));
        System.out.println(str1.compareTo(str3));
        System.out.println(str2.compareTo(str1));

    }
}
