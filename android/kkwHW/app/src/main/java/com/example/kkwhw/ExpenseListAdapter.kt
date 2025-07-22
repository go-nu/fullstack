package com.example.kkwhw

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.example.kkwhw.databinding.ItemExpenseBinding
import com.example.kkwhw.model.Expense

class ExpenseListAdapter(
    private val onDelete: (Expense) -> Unit
) : RecyclerView.Adapter<ExpenseListAdapter.ExpenseViewHolder>() {

    private var expenseList = listOf<Expense>()

    fun submitList(list: List<Expense>) {
        expenseList = list
        notifyDataSetChanged()
    }

    inner class ExpenseViewHolder(private val binding: ItemExpenseBinding) : RecyclerView.ViewHolder(binding.root) {
        fun bind(expense: Expense) {
            binding.textName.text = expense.name
            binding.textAmount.text = "${expense.amount}원"
            binding.textDate.text = expense.date
            binding.textCategory.text = expense.category
            binding.buttonDelete.setOnClickListener {
                onDelete(expense)
            }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ExpenseViewHolder {
        val binding = ItemExpenseBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return ExpenseViewHolder(binding)
    }

    override fun getItemCount() = expenseList.size

    override fun onBindViewHolder(holder: ExpenseViewHolder, position: Int) {
        holder.bind(expenseList[position])
    }
}
