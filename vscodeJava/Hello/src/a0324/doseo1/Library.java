package a0324.doseo1;

public class Library {
    private String title; // 책 제목
    private String author; // 책 저자
    private String location; // 책 위치
    private String isbn;
    private boolean avaiable;

    public Library() {
        // 기본 생성자
    }

    public Library(String title, String author, String location, String isbn) {
        this.title = title;
        this.author = author;
        this.location = location;
        this.isbn = isbn;
        this.avaiable = true;
    }
    
    public String getTitle() {
        return title;
    }
    public void setTitle(String title) {
        this.title = title;
    }
    public String getAuthor() {
        return author;
    }
    public void setAuthor(String author) {
        this.author = author;
    }
    public String getLocation() {
        return location;
    }
    public void setLocation(String location) {
        this.location = location;
    }
    public String getIsbn() {
        return isbn;
    }
    public void setIsbn(String isbn) {
        this.isbn = isbn;
    }
    public boolean isAvaiable() {
        return avaiable;
    }
    public void setAvaiable(boolean avaiable) {
        this.avaiable = avaiable;
    }

    @Override
    public String toString() {
        return "책 제목 : " + title + ", 저자 : " + author + ", 책 위치 : " + location
        + ", ISBN : " + isbn + ", 대출여부 : " + (avaiable ? "대출가능" : "대출불가능");
    }

    // 도서 대출 후 대출불가능 표시
    public void book() {
        this.avaiable = false;
    }

    
}
