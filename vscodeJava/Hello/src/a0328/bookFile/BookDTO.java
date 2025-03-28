package a0328.bookFile;

public class BookDTO {
    private String title;
    private String author;
    private String ISBN;
    private int price;
    
    public BookDTO() {
    }

    public BookDTO(String title, String author, String iSBN, int price) {
        this.title = title;
        this.author = author;
        ISBN = iSBN;
        this.price = price;
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
    public String getISBN() {
        return ISBN;
    }
    public void setISBN(String iSBN) {
        ISBN = iSBN;
    }
    public int getPrice() {
        return price;
    }
    public void setPrice(int price) {
        this.price = price;
    }

    @Override
    public String toString() {
        return "제목 : " + title + ", 저자 : " + author + ", ISBN : " + ISBN + ", 가격 : " + price + "원";
    }


}
