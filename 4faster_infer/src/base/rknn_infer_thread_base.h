#pragma once

#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "rknn_infer_base.h"


template<class OUTPUT, class JobAdditional=float>
class RknnInferThreadBase : public RknnInferBase<OUTPUT>
{

public:
    struct Job{
            cv::Mat input;
            OUTPUT output;
            std::shared_ptr<std::promise<OUTPUT>> pro;
            JobAdditional additional;

    };

    RknnInferThreadBase() { };
    virtual ~RknnInferThreadBase() {
        printf("thread over!\n");
        stop();
    }

    // void deep_copy_rknn(RknnInferThreadBase *p) {    // 拷贝，实现多线程的权重复用
    //     CHECK_RKNN(rknn_dup_context(&p->ctx_, &this->ctx_));
    //     this->io_num_ = p->io_num_;
    //     this->input_attrs_ = p->input_attrs_;
    //     this->output_attrs_ = p->output_attrs_;

    //     this->input_w_ = p->input_w_;
    //     this->input_w_ = p->input_w_;
    //     this->input_w_ = p->input_w_;

    //     this->out_scales_ = p->out_scales_;
    //     this->out_zps_ = p->out_zps_;

    //     init_io_tensor_mem();

    //     init_task_info();

    // }
    // virtual void init_task_info() = 0 ;              // 任务类初始化

     // 停止 由析构函数调用
    void stop(){
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
    }

    // 启动 初始化线程 用一个promise等待worker中的初始化结束
    void startup(){

        run_ = true;
        worker_ = std::make_shared<std::thread>(&RknnInferThreadBase::worker, this);
    }


    // 可以理解成生产者
    virtual std::shared_future<OUTPUT> commit(const cv::Mat &img) {
        Job job;
        job.pro = std::make_shared<std::promise<OUTPUT>>();
        if(!preprocess(job, img)){
            job.pro->set_value(OUTPUT());
            return job.pro->get_future();
        }
        
        ////////////////////图片放入队列////////////////////////////
        {
            std::unique_lock<std::mutex> l(jobs_lock_);
            jobs_.push(job);
        };
        cond_.notify_one();
        return job.pro->get_future();
    }

    // 获取任务 等待之前的任务执行完毕
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

    virtual bool preprocess(Job& job, const cv::Mat &img) = 0;
    virtual void worker() = 0 ;
    
    virtual OUTPUT infer(const cv::Mat &img) { return OUTPUT();}
  

protected:
    std::atomic<bool> run_;
    std::mutex jobs_lock_;
    std::shared_ptr<std::thread> worker_;
    std::condition_variable cond_;

    std::queue<Job> jobs_;
    Job fetch_job_;
};
