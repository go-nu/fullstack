package com.example.app0722

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.app0722.databinding.ItemEmotionBinding

// 감정 데이터와 RecycleView의 시각적 표현 사이를 연결
// RecycleView가 각 항복 뷰를 효율적으로 재활용할 수 있도록 EmotionViewHolder를 생성, 관리
class EmotionAdapter(
    // 삭제 버튼이 클릭 되었을 때 EmotionViewModel에게 해당 감정을 삭제하라는 onDelete 함수 제공
    private val onDelete: (Emotion) -> Unit
) : ListAdapter<Emotion, EmotionAdapter.EmotionViewHolder>(DiffCallback) {
    
    // 항목 업데이트 최적화
    inner class EmotionViewHolder(private val binding: ItemEmotionBinding) :
        RecyclerView.ViewHolder(binding.root) {
        fun bind(emotion: Emotion) {
            binding.tvDate.text = emotion.date
            binding.tvFeeling.text = emotion.feeling
            binding.tvContent.text = emotion.content
            binding.btnDelete.setOnClickListener { onDelete(emotion) }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): EmotionViewHolder {
        val binding = ItemEmotionBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return EmotionViewHolder(binding)
    }

    override fun onBindViewHolder(holder: EmotionViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    companion object {
        private val DiffCallback = object : DiffUtil.ItemCallback<Emotion>() {
            override fun areItemsTheSame(oldItem: Emotion, newItem: Emotion) = oldItem.id == newItem.id
            override fun areContentsTheSame(oldItem: Emotion, newItem: Emotion) = oldItem == newItem
        }
    }
}