function addToCart(bookid){
    if(confirm("장바구니에 도서를 추가합니다.") == true){
        document.addForm.action="/BookMarket/cart/book/" + bookid;
        document.addForm.submit();
    }
}
// 장바구니에 등록된 도서 항목을 삭제하는 메소드
function removeFromCart(bookid, cartId){
    document.removeForm.action="/BookMarket/cart/book/" + bookid;
    document.removeForm.submit();
    setTimeout('location.reload()', 10);
    // 바로 새로고침하면 남아있을 수 있어서 10ms 후 새로고침 하도록
}
function clearCart(cartId){
    if(confirm("모든 도서를 장바구니에서 삭제합니까?") == true){
        document.clearForm.submit();
        setTimeout('location.reload()', 10);
    }
}