package a0321.Account1;

public class Student {
    private String name;
    private int studentId;
    private int kor, math, eng;
    
    public void setName(String name) {
        this.name = name;
    }
    public void setStudentId(int studentId) {
        this.studentId = studentId;
    }
    public void setKor(int kor) {
        this.kor = kor;
    }
    public void setMath(int math) {
        this.math = math;
    }
    public void setEng(int eng) {
        this.eng = eng;
    }
    public String getName() {
        return name;
    }
    public int getStudentId() {
        return studentId;
    }

    public int getKor() {
        return kor;
    }

    public int getMath() {
        return math;
    }

    public int getEng() {
        return eng;
    }
    Student() {
        
    }
    Student(String n, int i, int k, int m, int e) {
        name = n;
        studentId = i;
        kor = k;
        math = m;
        eng = e;
    }
    public double getAverage() {
    // return (double)(kor + math + eng) / 3;
       return (kor + math + eng) / 3.0;
    }


}
