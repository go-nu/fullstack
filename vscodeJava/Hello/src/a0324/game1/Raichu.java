package a0324.game1;

public class Raichu extends Pikachu {
    
    public Raichu(int energy, String type, int level) {
        super(energy, type, level);
    }
    public Raichu(int energy, String type) {
        super(energy, type, 20);
    }

    @Override
    public String aAttack() {
        // return super.aAttack();
        return "백만볼트";
    }
    @Override
    public String bAttack() {
        return "볼트체인지";
    }

    public String cAttack() {
        if(level >= 40) return "아이언테일";
        else return "사용불가";
    }
    
}
