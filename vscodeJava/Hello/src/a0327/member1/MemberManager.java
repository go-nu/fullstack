package a0327.member1;

import java.util.ArrayList;

public class MemberManager {
    // private ArrayList<Member> members = new ArrayList<>();
    private ArrayList<Member> members;

    public MemberManager() {
        members = new ArrayList<>();
    }

    public void addMember(String newName, int newAge, String newEmail) {
        Member newMember = new Member(newName, newAge, newEmail);
        members.add(newMember);
    }

    public void displayAllMembers() {
        if(members.isEmpty()) System.out.println("등록된 회원이 없습니다.");
        else {
            System.out.println("전체 회원 목록> ");
            for(Member m : members) {
                System.out.println(m);
            }
        }
    }

    public Member findMember(String name) {
        for(Member m : members) {
            if (m.getName().equals(name)) return m;
        }
        return null;
    }

    // public void updateMember(String updateName, int newAge, String newEmail) {
    //     Member m = findMember(updateName);
    //     if(m != null) {
    //         m.setAge(newAge);
    //         m.setEmail(newEmail);
    //     } else {
    //         System.out.println("회원을 찾을 수 없습니다.");
    //     }
    // }

    public void updateMember(String updateName, int newAge, String newEmail) {
        for(int i = 0; i < members.size(); i++) {
            if (members.get(i).getName().equals(updateName)) {
                members.set(i, new Member(updateName, newAge, newEmail));
                return;
            }
        }
    }

    public void delMember(String delName) {
        Member m = findMember(delName);
        if(m != null) {
            members.remove(m);
        } else {
            System.out.println("회원을 찾을 수 없습니다.");
        }
    }



}
