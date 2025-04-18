package a0409.musicApp;

public class User {
    private String name;
    private int birth;
    private String id;
    private String pw;
    
    public User(String name, int birth, String id, String pw) {
        this.name = name;
        this.birth = birth;
        this.id = id;
        this.pw = pw;
    }
    
    public String getName() {
        return name;
    }
    public void setName(String name) {
        this.name = name;
    }
    public int getBirth() {
        return birth;
    }
    public void setBirth(int birth) {
        this.birth = birth;
    }
    public String getId() {
        return id;
    }
    public void setId(String id) {
        this.id = id;
    }
    public String getPw() {
        return pw;
    }
    public void setPw(String pw) {
        this.pw = pw;
    }
}
