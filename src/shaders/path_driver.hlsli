#ifndef QUANTILOOM_PATH_DRIVER_HLSLI
#define QUANTILOOM_PATH_DRIVER_HLSLI

// ============================================================================
// The path driver: raygen shades the vertices of a path itself
// ============================================================================
// Included by raygen only. It walks the tree the recursion walked, depth
// first and in the same order, with the vertex library the closest-hit stage
// runs (ShadeVertexA / Advance / CompleteChild / FoldVertex) and the sky
// evaluator the miss stage runs (ShadeMiss), so every draw is made from the
// stream state it was always made from.
//
// A ray traced from here is record-only (QlTraceRecord): closest-hit writes
// where it landed, miss says that it missed, and the vertex is shaded here.
// Both environment-bounce lobes are descended into the same way: the parent's
// frame is pushed, the child is shaded, and the parent resumes once the
// child's answer is in. Every other child -- transmission, dispersion, and the
// fusion interface's, which phase A traces itself -- is still traced through
// the hit stage, whose ShadeVertex shades that whole subtree recursively.
//
// One frame per raygen-shaded ancestor of the vertex being shaded. A vertex
// pushes only to take a bounce, and SampleBounce takes one only at depth <
// MAX_PATH_DEPTH. The bounce chain from the primary vertex is the only way
// down this stack, so the vertex at index i has depth i and at most
// MAX_PATH_DEPTH frames are live. Were that ever to stop holding, a bounce
// asked for with the stack full is traced through the hit stage instead --
// which is what happened to every bounce before, and gives the same answer.
//
// What a frame keeps beside the VertexFrame:
//   v       the vertex's own state as phase A left it, which the fold writes
//           its outputs into;
//   kind    the kind of the child in flight, which CompleteChild needs.
//
// The fusion medium statics are not kept: a child's phase A overwrites them
// (the recursion never noticed, since each closest-hit invocation had its own
// and QlTraceRadiance saved and restored them around its trace), and they are
// put back from v before a parent resumes -- see QlRestoreMedium.
//
// Every vertex runs on through ONE call of QlRunVertex, whether it was just
// shaded or is resuming after a child, so the fold and the draws after a child
// exist once in this stage. The vertex in hand lives in locals: a frame is
// written only when its vertex descends, and read once when it resumes.
// ============================================================================

#include "shade_vertex.hlsli"
#include "shade_miss.hlsli"

struct PathFrame {
    VertexFrame f;
    PathVertex  v;
    uint        kind;
};

bool QlDescendsInRaygen(uint kind) {
    return kind == QL_CHILD_BOUNCE || kind == QL_CHILD_BOUNCE_DT;
}

// Run a vertex on from where its frame stands, as ShadeVertex's loop does,
// until it asks for a child that raygen descends into (true, with the request
// in req) or it has folded (false, with its answer in v). Any other child is
// traced through the hit stage and folded in on the spot.
bool QlRunVertex(inout VertexFrame f, inout PathVertex v, inout ChildRequest req,
                 bool canDescend) {
    [loop] while (Advance(f, v.depth, req)) {
        if (canDescend && QlDescendsInRaygen(req.kind)) {
            return true;
        }
        CompleteChild(f, req.kind, QlTraceRadiance(req));
    }
    FoldVertex(f, v);
    return false;
}

// The fusion medium statics as v's phase A set them. A vertex's medium fields
// are written only while it is built (InheritFusionMedium, raygen's root, a
// fusion child's copy) and never while it is shaded, so they still hold what
// ShadeVertexA copied into the statics on entry.
void QlRestoreMedium(PathVertex v) {
    fusionActiveMediumCount = v.fusionMediumCount;
    [unroll]
    for (uint medium = 0; medium < kFusionMaxInitialMedia; ++medium)
        fusionActiveMedia[medium] = v.fusionMedia[medium];
}

// The whole path of one primary ray: vin is what the recursion's primary
// payload carried in, and the result is what it carried out.
PathVertex RunPath(RayDesc ray, uint rayFlags, PathVertex vin) {
    PathFrame stack[MAX_PATH_DEPTH];
    uint sp = 0u;
    PathVertex v = vin;   // the vertex in hand
    bool pathDone = false;
    [loop] while (!pathDone) {
        // The vertex at the end of `ray`, up to its first child. The hit ray
        // is the one the closest-hit stage reconstructs from its intrinsics.
        // f and req are declared after the trace so that nothing of the
        // previous vertex is live across it.
        HitRecord rec;
        const bool hit = QlTraceRecord(ray, rayFlags, v, rec);
        VertexFrame f = (VertexFrame)0;
        ChildRequest req = (ChildRequest)0;
        bool running = false;
        if (hit) {
            RayDesc hitRay = ray;
            hitRay.TMax = rec.t;
            ShadeVertexA(rec, hitRay, v, f, req);
            running = (f.ctl & QL_CTL_FINISHED) == 0u;
        } else {
            ShadeMiss(v, ray);
        }

        // Run the vertex in hand on until it asks for a bounce (push it and
        // descend) or is finished (fold it into its parent, which becomes the
        // vertex in hand), until the primary vertex is done.
        bool descend = false;
        [loop] while (!descend && !pathDone) {
            bool asked = false;
            if (running) {
                asked = QlRunVertex(f, v, req, sp < MAX_PATH_DEPTH);
            }
            if (asked) {
                stack[sp].f = f;
                stack[sp].v = v;
                stack[sp].kind = req.kind;
                ++sp;
                descend = true;
            } else if (sp == 0u) {
                pathDone = true;
            } else {
                --sp;
                const PathVertex child = v;
                f = stack[sp].f;
                v = stack[sp].v;
                QlRestoreMedium(v);
                CompleteChild(f, stack[sp].kind, child);
                running = true;
            }
        }

        if (descend) {
            // Built exactly as QlTraceRadiance builds it, the medium included:
            // the statics hold the parent's at this point.
            ray = req.ray;
            rayFlags = req.rayFlags;
            v = QlFromRay(QlChildPayload(req));
        }
    }
    return v;
}

#endif // QUANTILOOM_PATH_DRIVER_HLSLI
