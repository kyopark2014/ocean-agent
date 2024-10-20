# 인프라 설치하기

## Bedrock 사용 권한 설정하기

LLM으로 Anthropic의 Claude 3을 사용하기 위하여, Amazon Bedrock의 us-west-2, us-east-1, ca-central-1, eu-west-2, sa-east-1 리전을 사용합니다. 각 리전의 [Amazon Bedrock Console](https://console.aws.amazon.com/bedrock)의 [Model access]에 접속해서 [Edit]를 선택하여 "Anthropic Claude3 Sonnet"과 Titan Embedding V2을 enable 합니다.

![image](https://github.com/user-attachments/assets/f259bb17-cbd4-4f9e-8025-6552953a5899)



## CDK를 이용한 인프라 설치하기

여기서는 [AWS Cloud9](https://aws.amazon.com/ko/cloud9/)에서 [AWS CDK](https://aws.amazon.com/ko/cdk/)를 이용하여 인프라를 설치합니다. 

1) [Cloud9 Console](https://us-west-2.console.aws.amazon.com/cloud9control/home?region=us-west-2#/create)에 접속하여 [Create environment]-[Name]에서 “chatbot”으로 이름을 입력하고, EC2 instance는 “m5.large”를 선택합니다. 나머지는 기본값을 유지하고, 하단으로 스크롤하여 [Create]를 선택합니다.

![image](https://github.com/kyopark2014/stream-chatbot-for-amazon-bedrock/assets/52392004/c85c2ef5-4f96-4528-b5d4-ab9d3e52324e)

2) [Environment](https://us-west-2.console.aws.amazon.com/cloud9control/home?region=us-west-2#/)에서 “chatbot”를 [Open]한 후에 아래와 같이 터미널을 실행합니다.

![image](https://github.com/kyopark2014/stream-chatbot-for-amazon-bedrock/assets/52392004/fcf24f93-9ab3-4905-be8d-8146c7371951)

3) EBS 크기 변경

아래와 같이 스크립트를 다운로드 합니다. 

```text
curl https://raw.githubusercontent.com/kyopark2014/technical-summary/main/resize.sh -o resize.sh
```

이후 아래 명령어로 용량을 80G로 변경합니다.
```text
chmod a+rx resize.sh && ./resize.sh 80
```


4) 소스를 다운로드합니다.

```java
git clone https://github.com/kyopark2014/ocean-agent
```

5) cdk 폴더로 이동하여 필요한 라이브러리를 설치합니다.

```java
cd ocean-agent/cdk-ocean-agent/ && npm install
```

7) CDK 사용을 위해 Boostraping을 수행합니다.

아래 명령어로 Account ID를 확인합니다.

```java
aws sts get-caller-identity --query Account --output text
```

아래와 같이 bootstrap을 수행합니다. 여기서 "account-id"는 상기 명령어로 확인한 12자리의 Account ID입니다. bootstrap 1회만 수행하면 되므로, 기존에 cdk를 사용하고 있었다면 bootstrap은 건너뛰어도 됩니다.

```java
cdk bootstrap aws://[account-id]/us-west-2
```

8) 아래 명령어로 인프라를 설치합니다.

```java
cdk deploy --require-approval never --all
```

인프라가 설치가 되면 아래와 같은 Output을 확인할 수 있습니다. 

![noname](https://github.com/user-attachments/assets/21488aac-9319-4f80-bc7f-c2c855a68ac9)

9) Output의 HtmlUpdateCommend을 아래와 같이 복사하여 실행합니다.

![noname](https://github.com/user-attachments/assets/f7971246-3b38-441e-935c-b1ebfd5b3be9)

    
10) API에 대한 Credential을 획득하고 입력합니다.

- 일반 검색을 위하여 [Tavily Search](https://app.tavily.com/sign-in)에 접속하여 가입 후 API Key를 발급합니다. 이것은 tvly-로 시작합니다.

Tavily의 경우 1000건/월을 허용하므로 여러 건의 credential을 사용하면 편리합니다. 따라서, 아래와 같이 array형태로 입력합니다. 

```java
["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","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"]
```

- [langsmith.md](./langsmith.md)를 참조하여 [LangSmith](https://www.langchain.com/langsmith)에 가입후 API Key를 발급 받습니다.

[Secret manger](https://us-west-2.console.aws.amazon.com/secretsmanager/listsecrets?region=us-west-2)에 접속하여,  [tavilyapikey-writing-agent](https://us-west-2.console.aws.amazon.com/secretsmanager/secret?name=tavilyapikey-writing-agent&region=us-west-2), [langsmithapikey-writing-agent](https://us-west-2.console.aws.amazon.com/secretsmanager/secret?name=langsmithapikey-writing-agent&region=us-west-2)에 접속하여, [Retrieve secret value]를 선택 후, api key를 입력합니다.

Output의 WebUrlforstreamchatbot의 URL로 접속합니다. 만약 Credential을 입력 전에 URL을 접속을 했다면, Lambda를 재배포하거나 일정 시간후에 Lamba가 내려갈때까지 기다렸다가 재접속하여야 하므로, Credential들을 입력 후에 URL로 접속하는것이 좋습니다. 

11) RAG를 사용하기를 원하는 경우에 Amazon Bedrock Knowledge Base를 [knowledge-base.md](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/knowledge-base.md)에 따라 설정합니다. 여기에서는 knowledge base의 이름으로 "aws-rag"을 사용하고 있습니다. knowledge_base_id를 구하기 위하여 knowledge base의 이름을 이용하고 있으므로, knowledge base의 이름을 변경할 경우에는 [cdk-writing-agent-stack.ts](./cdk-writing-agent/lib/cdk-writing-agent-stack.ts)에서 "knowledge_base_name"을 수정후에 재배포합니다.
