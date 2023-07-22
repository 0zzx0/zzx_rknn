#pragma once
/**
 * @file rknn_infer_thread_base.hpp
 * @author zzx
 * @brief 生产者消费者多线程推理基类
 * @version 0.1
 * @date 2023-07-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "rknn_infer_base.hpp"


/**
 * @brief 生产者消费者推理基类
 * 
 * @tparam OUTPUT 输出类型
 * @tparam JobAdditional 备用自定义结构
 */
template<class OUTPUT, class JobAdditional=float>
class RknnInferThreadBase : public RknnInferBase<OUTPUT> {

public:

    /**
     * @brief 工作结构体
     * 
     */
    struct Job {
        cv::Mat input;
        OUTPUT output;
        std::shared_ptr<std::promise<OUTPUT>> pro;
        JobAdditional additional;
    };

    /**
     * @brief 销毁资源 清空工作队列 关闭消费者线程
     * 
     */
    virtual ~RknnInferThreadBase() {
        run_ = false;
        cond_.notify_all();

        // 清空工作队列
        {
            std::unique_lock<std::mutex> l(jobs_lock_);
            while(!jobs_.empty()){
                auto& item = jobs_.front();
                if(item.pro)
                    item.pro->set_value(OUTPUT());
                jobs_.pop();
            }
        };

        if(worker_->joinable()){
            worker_->join();
            worker_.reset();
        }
        printf("RknnInferThreadBase release!\n");
    }


    /**
     * @brief 初始化worker线程 即检测线程
     * 
     */
    void startup(){

        run_ = true;
        worker_ = std::make_shared<std::thread>(&RknnInferThreadBase::worker, this);
    }


    /**
     * @brief 提交图片任务 生产者
     * 
     * @param img 检测图片
     * @return std::shared_future<OUTPUT> 返回std::future的OUTPUT
     */
    virtual std::shared_future<OUTPUT> commit(const cv::Mat &img) {
        Job job;
        job.pro = std::make_shared<std::promise<OUTPUT>>();
        // 预处理
        if(!preprocess(job, img)){
            job.pro->set_value(OUTPUT());
            return job.pro->get_future();
        }
        ////////////////////上锁提交图片到任务队列//////////////////////
        {
            std::unique_lock<std::mutex> l(jobs_lock_);
            jobs_.push(job);
        };
        cond_.notify_one();
        return job.pro->get_future();
    }

    /**
     * @brief 从任务队列取图片
     * 
     * @return true 获取成功
     * @return false 获取失败
     */
    virtual bool get_job_and_wait(){

        std::unique_lock<std::mutex> l(jobs_lock_);
        cond_.wait(l, [&](){
            return !run_ || !jobs_.empty();
        });

        if(!run_) return false;
        
        fetch_job_ = std::move(jobs_.front());
        jobs_.pop();
        return true;
    }

    /**
     * @brief 预处理。纯虚函数
     * 
     * @param job 工作
     * @param img 图片
     * @return true 处理成功
     * @return false 处理失败
     */
    virtual bool preprocess(Job& job, const cv::Mat &img) = 0;

    /**
     * @brief 工作线程。纯虚函数
     * 
     */
    virtual void worker() = 0 ;
    
    /**
     * @brief 将父类纯虚函数转成普通虚函数。后续子类不再使用
     * 
     * @param img 
     * @return OUTPUT 
     */
    virtual OUTPUT infer(const cv::Mat &img) { return OUTPUT();}
  

protected:
    std::atomic<bool> run_;     // 运行状态
    std::mutex jobs_lock_;      // 队列锁
    std::condition_variable cond_;          // 条件变量
    std::shared_ptr<std::thread> worker_;   // 工作线程
    
    std::queue<Job> jobs_;  // 工作队列
    Job fetch_job_;         // 当前工作
};
