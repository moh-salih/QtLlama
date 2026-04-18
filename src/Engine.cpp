#include <QtLlama/Engine.h>
#include <QtLlama/LlamaBackend.h>

#include <QDebug>


namespace {
using namespace QtLlama;
// Helper 
Formatter autoDetect(llama_model* model) {
    const char* tmpl_ptr = llama_model_chat_template(model, nullptr);
    if (!tmpl_ptr) {
        qWarning() << "QtLlama: Model has no embedded chat template. "
                      "Set Config::promptFormatter explicitly.";
        // Return a formatter that just passes the last user message through.
        // This is a last-resort fallback — set promptFormatter explicitly for best results.
        return [](const QList<Message>& messages) -> QString {
            for (int i = messages.size() - 1; i >= 0; --i) {
                if (messages[i].role == Role::User)
                    return messages[i].content;
            }
            return {};
        };
    }

    qInfo() << "QtLlama: Using embedded chat template from model metadata.";

    std::string tmpl(tmpl_ptr);
    return [tmpl](const QList<Message>& messages) -> QString {
        std::vector<llama_chat_message> cMessages;
        std::vector<std::string> contentStrings;

        contentStrings.reserve(messages.size());
        for (const Message& msg : messages)
            contentStrings.push_back(msg.content.toStdString());

        for (size_t i = 0; i < (size_t)messages.size(); ++i) {
            const char* roleStr = "user";
            if (messages[i].role == Role::System)     roleStr = "system";
            else if (messages[i].role == Role::Assistant) roleStr = "assistant";
            cMessages.push_back({ roleStr, contentStrings[i].c_str() });
        }

        int required = llama_chat_apply_template(
            tmpl.c_str(),
            cMessages.data(),
            cMessages.size(),
            true,
            nullptr, 0
        );

        if (required < 0) {
            qWarning() << "QtLlama: llama_chat_apply_template failed. "
                          "Returning last user message as fallback.";
            for (int i = messages.size() - 1; i >= 0; --i)
                if (messages[i].role == Role::User)
                    return messages[i].content;
            return {};
        }

        std::vector<char> buf(required + 1, '\0');
        llama_chat_apply_template(
            tmpl.c_str(),
            cMessages.data(),
            cMessages.size(),
            true,
            buf.data(),
            static_cast<int>(buf.size())
        );

        return QString::fromUtf8(buf.data());
    };
}

    
}


namespace QtLlama {


Engine::Engine(QObject *parent) : IEngine(parent) {
    QtLlama::ensureBackendInit();
}

Engine::~Engine() {
    unloadModel();
}

void Engine::setConfig(QSharedPointer<Config> config) {
    mConfig = config;
}

void Engine::loadModel() {
    unloadModel();
    emit modelStatusChanged(Status::Loading);

    if (!mConfig || mConfig->modelPath.isEmpty()) {
        emit errorOccurred("LLM model path is not configured.");
        emit modelStatusChanged(Status::Error);
        return;
    }
    

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = mConfig->nGpuLayers;
    m_model = llama_model_load_from_file(mConfig->modelPath.toStdString().c_str(), mp);

    if (!m_model) {
        emit errorOccurred("Failed to load LLM model file");
        emit modelStatusChanged(Status::Error);
        return;
    }


    mPromptFormatter = autoDetect(m_model);


    llama_context_params cp = llama_context_default_params();
    cp.n_threads       = mConfig->threadCount;
    cp.n_batch         = mConfig->batchSize;
    cp.n_threads_batch = mConfig->batchThreads;
    cp.n_ctx           = mConfig->contextLength == 0 ? llama_model_n_ctx_train(m_model) : mConfig->contextLength;

    m_ctx = llama_init_from_model(m_model, cp);
    if (!m_ctx) {
        emit errorOccurred("Failed to create llama context.");
        emit modelStatusChanged(Status::Error);
        return;
    }

    m_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(m_sampler, llama_sampler_init_top_k(mConfig->topK));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_top_p(mConfig->topP, 1));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_temp(mConfig->temperature));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_penalties(
        mConfig->repeatPenaltyLastN,                 
        mConfig->repeatPenalty,   
        mConfig->penaltyFreq,                     
        mConfig->penaltyPresent
    ));

    llama_sampler_chain_add(m_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    emit modelStatusChanged(Status::Ready);
}

void Engine::unloadModel() {
    m_abort.store(true);
    if (m_sampler) { llama_sampler_free(m_sampler); m_sampler = nullptr; }
    if (m_ctx)     { llama_free(m_ctx);              m_ctx     = nullptr; }
    if (m_model)   { llama_model_free(m_model);      m_model   = nullptr; }

    emit modelStatusChanged(Status::Idle);
}

QString Engine::applyPromptFormat(const QList<Message>& messages) const {
    return mPromptFormatter(messages);
}

void Engine::generate(const QList<Message>& messages) {
    if (!m_ctx || !m_model || !m_sampler) return;

    emit isGeneratingChanged(true);
    m_abort.store(false);

    // 1. Format messages into a single prompt string.
    const QString formatted = applyPromptFormat(messages);

    // 2. Clear KV cache.
    llama_memory_t mem = llama_get_memory(m_ctx);
    if (mem) llama_memory_clear(mem, true);

    // 3. Tokenize.
    const llama_vocab* vocab = llama_model_get_vocab(m_model);
    std::string text = formatted.toStdString();
    std::vector<llama_token> tokens(text.size() + 32);

    int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);
    if (n < 0) {
        emit errorOccurred("Failed to tokenize prompt.");
        emit isGeneratingChanged(false);
        return;
    }
    tokens.resize(n);

    // 4. Fill and submit the initial batch.
    llama_batch batch = llama_batch_init(mConfig->batchSize, 0, 1);
    for (int i = 0; i < n; ++i) {
        batch.token[i]      = tokens[i];
        batch.pos[i]        = i;
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = 0;
        batch.logits[i]     = (i == n - 1);
    }
    batch.n_tokens = n;

    if (llama_decode(m_ctx, batch) != 0) {
        emit errorOccurred("Initial llama_decode failed.");
        llama_batch_free(batch);
        emit isGeneratingChanged(false);
        return;
    }

    // 5. Generation loop.
    int cur        = n;
    int tokenCount = 0;
    QString fullResponse;
   
    while (!m_abort.load()) {
        if (mConfig->maxTokens > 0 && tokenCount >= mConfig->maxTokens) break;

        llama_token id = llama_sampler_sample(m_sampler, m_ctx, -1);
        llama_sampler_accept(m_sampler, id);

        if (llama_vocab_is_eog(vocab, id)) break;

        char buf[128];
        int len = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (len > 0) {
            QString token = QString::fromUtf8(buf, len);
            fullResponse += token;
            emit tokenGenerated(token);
            tokenCount++;
        }

        batch.n_tokens  = 1;
        batch.token[0]  = id;
        batch.pos[0]    = cur++;
        batch.logits[0] = true;

        if (llama_decode(m_ctx, batch) != 0) {
            emit errorOccurred("Inference decode failed.");
            break;
        }
    }

    llama_batch_free(batch);


    emit responseReady(fullResponse);
    emit isGeneratingChanged(false);
}

void Engine::stop()  { m_abort.store(true);  }
void Engine::reset() { m_abort.store(false); }















} // namespace QtLlama
