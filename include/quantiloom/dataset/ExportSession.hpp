#pragma once

#include "core/Config.hpp"
#include "core/Image.hpp"
#include "core/Platform.hpp"
#include "io/ImageIO.hpp"

#include <memory>

namespace quantiloom::dataset {

/// Versioned JSON document captured when a product is generated. The SDK owns
/// its schema; hosts transport it without reconstructing renderer state.
struct RenderProvenance {
    String json;
};

struct VerificationReport {
    bool valid = false;
    String json;
};

/// Transaction for one output stem. Staging and publication stay on the same
/// filesystem. Per-artifact reservations reject overlapping output sets even
/// across subdirectories; disjoint sets can render and publish concurrently.
/// Claims survive a crash (inspect and remove stale *.quantiloom-export.lock
/// directories explicitly). StagingPath reserves its destination until destruction.
/// Destruction before Commit leaves the previous sample intact. Commit retains
/// previous files until success and rolls them back on publication failure.
/// Readers may see an unavailable record during commit. If recovery itself fails,
/// private backups and claims are retained and their location is reported.
class QL_API ExportSession {
public:
    static Result<std::unique_ptr<ExportSession>, String> Create(
        const String& outputPath, const Config& replayConfig,
        const RenderProvenance& provenance);
    ~ExportSession();
    ExportSession(const ExportSession&) = delete;
    ExportSession& operator=(const ExportSession&) = delete;

    /// Name is a relative artifact path; traversal and symlink parents are rejected. Products must have unique IDs and names.
    /// Description is frozen product metadata, not the current renderer state.
    Result<void, String> WriteImage(const String& name, const String& productId,
                                  const Image& image, const String& descriptionJson);
    Result<void,String> WriteUIntImage(const String& name,const String& productId,
                                      const UIntImage& image,const String& descriptionJson);
    /// Reserve a staged path for a streaming writer, then register it only after
    /// that writer closes successfully. Unregistered files are never published.
    /// The path is valid only for this session's lifetime. On Linux it uses a
    /// /proc/self/fd directory capability and must not be canonicalized or passed
    /// to another process. Windows holds non-deletable, non-reparse ancestors.
    Result<String, String> StagingPath(const String& name) const;
    Result<void, String> RegisterFile(const String& name, const String& productId,
                                    const String& descriptionJson);
    Result<void, String> Commit();
    [[nodiscard]] const String& RecordId() const;
    [[nodiscard]] const String& SidecarName() const;

    /// Checks the version, state, product identities and all managed hashes.
    /// Resource/environment verification is separately reported by provenance;
    /// an intact export alone does not establish reproducibility.
    static VerificationReport Verify(const String& recordPath);

private:
    ExportSession();
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace quantiloom::dataset
