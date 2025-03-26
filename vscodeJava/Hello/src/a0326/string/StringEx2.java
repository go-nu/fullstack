package a0326.string;

public class StringEx2 {
    public static void main(String[] args) {
        String str = new String("abcd");
        System.out.println("원본 문자열 : " + str);

        // 첫 글자의 유니코드 비교 (원본 첫글자의 유니코드 - 비교 첫글자의 유니코드)
        System.out.println(str.compareTo("bcdf")); // 다르면 -1
        System.out.println(str.compareTo("abcd")); // 같으면 0 출력
        
        System.out.println(str.compareTo("Abcd")); // 32
        System.out.println(str.compareToIgnoreCase("Abcd")); // 0
        System.out.println("compareTo() 메소드 호출 후 원본 문자열 : " + str);

    }
}
