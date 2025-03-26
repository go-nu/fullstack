package a0326.string;

public class StrBuffer {
    public static void main(String[] args) {
        StringBuffer str = new StringBuffer("java");
        System.out.println("원본 문자열 : " + str);
        
        System.out.println(str.append("수업"));
        System.out.println("append 메소드 호출 문자열 : " + str);

    }
}
