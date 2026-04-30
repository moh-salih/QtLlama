#pragma once
#include <QObject>
#include <QString>
#include <QList>
#include <QMetaType>
#include <vector>
#include <functional>

namespace QtLlama {
    Q_NAMESPACE

    enum class Status {
        Idle,
        Loading,
        Ready,
        Error
    };
    Q_ENUM_NS(Status);

    enum class Error {
        ModelPathEmpty,
        ModelLoadFailed,
        ContextInitFailed,
        TokenizationFailed,
        DecodeFailed,
        EmbeddingRetrieveFailed
    };
    Q_ENUM_NS(Error);

    enum class Role {
        System,
        User,
        Assistant
    };
    Q_ENUM_NS(Role);

    struct Message {
        Role    role;
        QString content;
    };

    struct Config {
        QString         modelPath;
        int             threadCount             = 1;
        int             batchThreads            = 2;
        int             contextLength           = 0;
        int             batchSize               = 2048;
        float           temperature             = 0.7f;
        float           topP                    = 0.9f;
        int             topK                    = 40;
        int             maxTokens               = -1;
        int             nGpuLayers              = 0;
        float           repeatPenalty           = 1.1f;
        int             repeatPenaltyLastN      = 64;
        float           penaltyFreq             = 0.0f;
        float           penaltyPresent          = 0.0f;
        bool            autoReload              = true;
    };

    struct EmbedConfig {
        QString         modelPath;
        int             nCtx                    = 2048;
        int             nThreads                = 1;
        int             nGpuLayers              = 0;
        bool            autoReload              = true;
    };

    inline QString errorToString(Error error) {
        switch (error) {
            case Error::ModelPathEmpty:          return "Model path is not configured.";
            case Error::ModelLoadFailed:         return "Failed to load model file.";
            case Error::ContextInitFailed:       return "Failed to initialize llama context.";
            case Error::TokenizationFailed:      return "Failed to tokenize prompt.";
            case Error::DecodeFailed:            return "Llama decode failed during inference.";
            case Error::EmbeddingRetrieveFailed: return "Could not retrieve embedding vector.";
            default:                             return "Unknown error.";
        }
    }

} // namespace QtLlama

#ifndef Q_META_TYPE_STD_VECTOR_FLOAT_DECLARED
#define Q_META_TYPE_STD_VECTOR_FLOAT_DECLARED
    Q_DECLARE_METATYPE(std::vector<float>)
#endif

Q_DECLARE_METATYPE(QtLlama::Message)
Q_DECLARE_METATYPE(QList<QtLlama::Message>)
