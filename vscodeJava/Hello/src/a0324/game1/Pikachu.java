package a0324.game1;

public class Pikachu {
    public int energy;
    public String type;
    public int level;

    public Pikachu() {
        this(100, "전기");
    }

    public Pikachu(int energy, String type) {
        this(energy, type, 1);
    }

    public Pikachu(int energy, String type, int level) {
        this.energy = energy;
        this.type = type;
        this.level = level;
    }

    public String aAttack() {
        return "십만볼트";
    }
    public String bAttack() {
        return "전광석화";
    }


}
