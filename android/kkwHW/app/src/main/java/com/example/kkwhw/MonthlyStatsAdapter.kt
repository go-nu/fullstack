package com.example.kkwhw

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.example.kkwhw.databinding.ItemMonthlyBinding
import com.example.kkwhw.model.MonthlyTotal

class MonthlyStatsAdapter : RecyclerView.Adapter<MonthlyStatsAdapter.MonthlyViewHolder>() {
    private var data = listOf<MonthlyTotal>()

    fun submitList(list: List<MonthlyTotal>) {
        data = list
        notifyDataSetChanged()
    }

    inner class MonthlyViewHolder(private val binding: ItemMonthlyBinding) : RecyclerView.ViewHolder(binding.root) {
        fun bind(item: MonthlyTotal) {
            binding.textMonth.text = item.date
            binding.textMonthTotal.text = "총 지출: ${item.total}원"
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MonthlyViewHolder {
        val binding = ItemMonthlyBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return MonthlyViewHolder(binding)
    }

    override fun getItemCount() = data.size

    override fun onBindViewHolder(holder: MonthlyViewHolder, position: Int) {
        holder.bind(data[position])
    }
}