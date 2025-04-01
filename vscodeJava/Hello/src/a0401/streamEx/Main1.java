package a0401.streamEx;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class Main1 {
    public static void main(String[] args) {
        Trader raoul = new Trader("Raoul", "Cambridge");
        Trader mario = new Trader("Mario", "Milan");
        Trader alan = new Trader("Alan", "Cambridge");
        Trader brian = new Trader("Brian", "Cambridge");
        List<Transaction> transactions = Arrays.asList(
            new Transaction(brian, 2011, 300),
            new Transaction(raoul, 2012, 1000),
            new Transaction(raoul, 2011, 400),
            new Transaction(mario, 2012, 700),
            new Transaction(mario, 2012, 700),
            new Transaction(alan, 2012, 950)
        );
        // 거래자, 거주지, 거래년도, 거래금액

        // 1. 2011년에 일어난 모든 거래를 찾아 오름차순으로 정렬
        practice1(transactions);
        System.out.println();
        // 2. 거래자가 근무하는 모든 도시(중복 없이) 나열
        practice2(transactions);
        System.out.println();
        // 3. Cambridge에서 근무하는 모든 거래자를 찾아 이름순 나열
        practice3(transactions);
        System.out.println();
        // 4. Milan에 거래자가 있는지
        practice4(transactions);
        System.out.println();
        // 5. Cambridge에 거주하는 거래자의 모든 Transaction 출력
        practice5(transactions);
        System.out.println();
        // 6. 모든 거래자의 이름을 알파뱃 순으로 정렬
        practice6(transactions);
        System.out.println();
        // 7. 최대값 구하기
        practice7(transactions);
        System.out.println();
        // 8. 최소값 구하기
        practice8(transactions);
    }

    private static void practice8(List<Transaction> transactions) {
        Optional<Transaction> result = transactions.stream()
            .min(Comparator.comparing(Transaction::getValue));
        System.out.println(result); // null이어도 Optional로 감싼 형태로 반환
        System.out.println(result.get().getValue());
    }

    private static void practice7(List<Transaction> transactions) {
        Transaction result = transactions.stream()
            .max(Comparator.comparing(Transaction::getValue))
            .orElse(null); // 값이 없으면 null
        System.out.println(result);
    }

    private static void practice6(List<Transaction> transactions) {
        List<String> result = transactions.stream()
            .map(Transaction::getTrader)
            .map(Trader::getName)
            .distinct()
            .sorted()
            .collect(Collectors.toList());
        System.out.println(result);
    }   

    private static void practice5(List<Transaction> transactions) {
        List<Transaction> result = transactions.stream()
            .filter(tran -> "Cambridge".equals(tran.getTrader().getCity()))
            .collect(Collectors.toList());
        System.out.println(result);
    }

    private static void practice4(List<Transaction> transactions) {
        boolean result = transactions.stream()
            .anyMatch(tran -> "Milan".equals(tran.getTrader().getCity()));
        System.out.println(result);
    }

    private static void practice3(List<Transaction> transactions) {
        List<Trader> result = transactions.stream()
            .map(Transaction::getTrader) // Transaction 객체에서 Trader만 가져옴
            .filter(trader -> "Cambridge".equals(trader.getCity()))
            .distinct()
            .sorted(Comparator.comparing(Trader::getName))
            .collect(Collectors.toList());
        System.out.println(result);
    }

    private static void practice2(List<Transaction> transactions) {
        List<String> result = transactions.stream()
            .map(tran -> tran.getTrader().getCity())
            .distinct()
            .collect(Collectors.toList());
        System.out.println(result);
    }

    private static void practice1(List<Transaction> transactions) {
        List<Transaction> result =  transactions.stream()
            .filter(transaction -> 2011 == transaction.getYear())
            .sorted(Comparator.comparing(Transaction::getValue)) // value 기준 정렬
            //.sorted((tran1, tran2) -> Integer.compare(tran1.getValue(), tran2.getValue()))
            // .sorted(Comparator.comparing(Transaction::getValue).reversed()) // value 기준 내림차순
            // .sorted(Comparator.comparing(transaction -> transaction.getTrader().getName())) // 이름 기준 정렬
            .collect(Collectors.toList());
        System.out.println(result);
    }
}
