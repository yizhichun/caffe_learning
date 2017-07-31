#include <vector>
#include <iostream>
#include <caffe/net.hpp>
using namespace caffe;
using namespace std;
int main(void)
{
	std::string proto("deploy.prototxt");
	Net<float> nn(proto, caffe::TEST);
	vector<string> bn = nn.blob_names();
	vector<string> ln = nn.layer_names();
	for(int i=0;i<bn.size();i++)
	{
		cout<<"Blob #"<<i<<":"<<bn[i]<<endl;
	}
	for(int i=0;i<ln.size();i++)
	{
		cout<<"Layer #"<<i<<":"<<ln[i]<<endl;
	}
	
	vector<Blob<float>*> learnable_params=nn.learnable_params();
	for(int i=0;i<learnable_params.size();i++){
                cout<<"Blob "<<i<<":"<<endl;
		Blob<float>* a=learnable_params[i];
                for(int u=0;u<a->num();u++){
                for(int v=0;v<a->channels();v++){
                for(int w=0;w<a->height();w++){
                for(int x=0;x<a->width();x++){
                        cout<<"learnalbe["<<u<<"]["<<v<<"]["<<w<<"]["<<x<<"]="<<a->data_at(u,v,w,x)<<endl;
                }}}}
        }
		
	Blob<float> a;
        cout<<"Size:"<<a.shape_string()<<endl;
        a.Reshape(1,1,4,4);
        cout<<"Size:"<<a.shape_string()<<endl;

        float* p = a.mutable_cpu_data();
        float* q = a.mutable_cpu_diff();
        for(int i=0;i<a.count();i++)
        {
                p[i]=i;
                q[i]=a.count() - 1 - i;
        }
	
	vector<Blob<float>*> data;
	vector<Blob<float>*> net_output_blobs;
	data.push_back(&a);
	net_output_blobs=nn.Forward(data);
	for(int i=0;i<net_output_blobs.size();i++){
		cout<<"Blob "<<i<<":"<<endl;
		Blob<float>* a=net_output_blobs[i];
		for(int u=0;u<a->num();u++){
        	for(int v=0;v<a->channels();v++){
        	for(int w=0;w<a->height();w++){
        	for(int x=0;x<a->width();x++){
                	cout<<"output["<<u<<"]["<<v<<"]["<<w<<"]["<<x<<"]="<<a->data_at(u,v,w,x)<<endl;
        	}}}}	
	}
	//cout<<nn.ForwardFrom(0)<<endl;
	return 0;
}
