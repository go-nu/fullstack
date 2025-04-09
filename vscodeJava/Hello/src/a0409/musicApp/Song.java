package a0409.musicApp;

public class Song {
    private String title;
    private String genre;
    private String artist;
    
    public String getTitle() {
        return title;
    }
    public void setTitle(String title) {
        this.title = title;
    }
    public String getGenre() {
        return genre;
    }
    public void setGenre(String genre) {
        this.genre = genre;
    }
    public String getArtist() {
        return artist;
    }
    public void setArtist(String artist) {
        this.artist = artist;
    }
    
    @Override
    public String toString() {
        return title + "(" + artist + ")" + "\t[" + genre + "]";
    }
    
}
