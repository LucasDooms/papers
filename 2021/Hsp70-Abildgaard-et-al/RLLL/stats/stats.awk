BEGIN{
    max=-100;
    sum=0;
    n=0;
}
{
    score=$2;
    if(score>max){max=score};
    sum+=score;
    n++;
}
END{
    print(max,sum/n)
}