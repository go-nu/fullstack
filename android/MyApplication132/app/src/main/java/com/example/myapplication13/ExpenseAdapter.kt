package com.example.myapplication13

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView

// RecycleView를 사용해 지출(Expense) 리스트를 화면에 표시
class ExpenseAdapter : ListAdapter<Expense, ExpenseAdapter.ExpenseViewHolder>(DiffCallback()) {
// ListAdapter<Expense, ExpenseAdapter.ExpenseViewHolder> 상속받아 리스트 변경시 UI 갱신

    inner class ExpenseViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvTitle = itemView.findViewById<TextView>(R.id.tv_title)
        private val tvAmount = itemView.findViewById<TextView>(R.id.tv_amount)
        private val tvCategory = itemView.findViewById<TextView>(R.id.tv_category)
        private val tvDate = itemView.findViewById<TextView>(R.id.tv_date)
        //bind 데이터와 UI 요소(View)를 연결
        fun bind(expense: Expense) {
            tvTitle.text = expense.title // 지출항목의 제목을 tvTitle 텍스트에 표시
            tvAmount.text = "₩ ${expense.amount}"
            tvCategory.text = expense.category
            tvDate.text = expense.date
        }
    }
    // RecycleView가 화면에 보여줄 ItemView를 새로 만들어야 할 때 호출 되는 함수
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ExpenseViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_expense, parent, false)
        return ExpenseViewHolder(view)
    }

    // RecycleView 에서 각 아이템의 데이터를 실제 View에 연결
    override fun onBindViewHolder(holder: ExpenseViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    // 두 아이템이 같은 객체인지 판단해서 변경된것만 업데이트
    class DiffCallback : DiffUtil.ItemCallback<Expense>() {
        override fun areItemsTheSame(oldItem: Expense, newItem: Expense): Boolean {
            return oldItem.id == newItem.id
        }

        override fun areContentsTheSame(oldItem: Expense, newItem: Expense): Boolean {
            return oldItem == newItem
        }
    }
}

// RecycleView 어댑터 클래스
// 지출 목록을 화면에 보여줄 때 각 아이템 뷰를 생성하고 바인딩함
// 예) 날짜별 지출 내역 목록 출력