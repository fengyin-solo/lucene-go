package index

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/geange/lucene-go/core/interface/index"
	"github.com/geange/lucene-go/core/store"
)

func TestLiveDocs(reader index.CodecReader) error {
	numDocs := reader.NumDocs()
	if reader.HasDeletions() {
		liveDocs := reader.GetLiveDocs()
		if liveDocs == nil {
			return errors.New("segment should have deletions, but liveDocs is null")
		}

		numLive := 0
		size := int(liveDocs.Len())
		for i := 0; i < size; i++ {
			if liveDocs.Test(uint(i)) {
				numLive++
			}
		}
		if numLive != numDocs {
			return errors.New("liveDocs count mismatch")
		}
		return nil
	}

	liveDocs := reader.GetLiveDocs()
	if liveDocs != nil {
		// it's ok for it to be non-null here, as long as none are set right?
		size := int(liveDocs.Len())
		for i := 0; i < size; i++ {
			if !liveDocs.Test(uint(i)) {
				return errors.New("liveDocs mismatch")
			}
		}
	}
	return nil
}

// SegmentStatus is the deterministic health status of a single segment.
type SegmentStatus uint8

const (
	// SegmentStatusOK means every check of the segment succeeded.
	SegmentStatusOK SegmentStatus = iota
	// SegmentStatusFailed means at least one check of the segment failed,
	// but the remaining segments were still inspected.
	SegmentStatusFailed
	// SegmentStatusCancelled means the segment was not inspected because
	// the check context was cancelled.
	SegmentStatusCancelled
)

// String returns the stable wire representation of the segment status.
func (s SegmentStatus) String() string {
	switch s {
	case SegmentStatusOK:
		return "OK"
	case SegmentStatusFailed:
		return "FAILED"
	case SegmentStatusCancelled:
		return "CANCELLED"
	default:
		return "UNKNOWN"
	}
}

// MarshalJSON serializes the status as a stable, upper-case string so that
// serialized reports never depend on internal enum numbers.
func (s SegmentStatus) MarshalJSON() ([]byte, error) {
	return json.Marshal(s.String())
}

// Segment check names. They are part of the serialized report contract.
const (
	segmentCheckReaderAccess = "readerAccess"
	segmentCheckLiveDocs     = "liveDocs"
	segmentCheckSegmentID    = "segmentId"
)

// SegmentCheckResult is the outcome of a single, named check for one segment.
// Messages only carry public evidence: the segment name, the check name and
// integer counts. File paths, raw identifiers and underlying IO errors are
// never exposed.
type SegmentCheckResult struct {
	Name    string `json:"name"`
	Passed  bool   `json:"passed"`
	Message string `json:"message,omitempty"`
}

// SegmentReport is the structured health information of one segment.
// Segments are always reported in commit order.
type SegmentReport struct {
	// Name is the segment name, e.g. "_0".
	Name string `json:"name"`

	// DocBase is the starting document number of the segment in the
	// top-level index, computed from cumulative segment maxDoc values.
	DocBase int `json:"docBase"`

	// MaxDoc is one greater than the largest document number in the segment,
	// including deleted documents.
	MaxDoc int `json:"maxDoc"`

	// LiveDocs is the number of live (non hard-deleted) documents.
	LiveDocs int `json:"liveDocs"`

	// DeletedDocs is the number of hard-deleted documents in the segment.
	DeletedDocs int `json:"deletedDocs"`

	// SoftDeletedDocs is the number of soft-deleted documents that are not
	// also hard-deleted.
	SoftDeletedDocs int `json:"softDeletedDocs"`

	// Status is the aggregated status of the segment checks.
	Status SegmentStatus `json:"status"`

	// Error is the safe, human readable summary of the first failure.
	// It is empty when Status is SegmentStatusOK.
	Error string `json:"error,omitempty"`

	// Checks is the ordered list of per-check results for the segment.
	Checks []SegmentCheckResult `json:"checks"`
}

// IndexReport is the structured, serializable health report of an index.
type IndexReport struct {
	// CommitGeneration is the generation of the inspected segments_N commit.
	CommitGeneration int64 `json:"commitGeneration"`

	// MaxDoc is the total number of documents including deleted ones.
	MaxDoc int `json:"maxDoc"`

	// NumDocs is the total number of live documents.
	NumDocs int `json:"numDocs"`

	// TotalDeletedDocs is the total number of hard-deleted documents.
	TotalDeletedDocs int `json:"totalDeletedDocs"`

	// TotalSoftDeletedDocs is the total number of soft-deleted documents.
	TotalSoftDeletedDocs int `json:"totalSoftDeletedDocs"`

	// SegmentCount is the number of segments declared by the commit.
	SegmentCount int `json:"segmentCount"`

	// Healthy is true when the commit is readable and every segment passed.
	Healthy bool `json:"healthy"`

	// Error carries the safe summary of a commit-level failure.
	Error string `json:"error,omitempty"`

	// Segments is the per-segment report ordered exactly as the commit.
	Segments []*SegmentReport `json:"segments"`
}

// CheckIndex performs read-only integrity checks against an index and builds
// a structured health report. It never writes, deletes or repairs index files
// and never closes a reader supplied by the caller.
type CheckIndex struct {
	dir store.Directory
}

// NewCheckIndex creates a read-only checker bound to the given directory.
func NewCheckIndex(dir store.Directory) *CheckIndex {
	return &CheckIndex{dir: dir}
}

// CheckIndexOptions configures a CheckIndex run.
type CheckIndexOptions struct {
	// Context carries cancellation. A nil context is replaced by
	// context.Background.
	Context context.Context

	// Reader optionally reuses an already opened DirectoryReader. When set,
	// the checker inspects that reader and never closes it. When nil, the
	// checker opens its own per-segment readers from the configured Directory
	// and releases them when the run finishes.
	Reader index.DirectoryReader
}

// NewCheckIndexOptions creates default options: background context and no
// pre-opened reader.
func NewCheckIndexOptions() *CheckIndexOptions {
	return &CheckIndexOptions{Context: context.Background()}
}

// DoCheck runs the configured checks and returns a deterministic report.
// A non-nil report is returned even when the returned error is non-nil, for
// example when the context is cancelled or no readable commit exists.
func (c *CheckIndex) DoCheck(opts *CheckIndexOptions) (*IndexReport, error) {
	if opts == nil {
		opts = NewCheckIndexOptions()
	}
	ctx := opts.Context
	if ctx == nil {
		ctx = context.Background()
	}

	if opts.Reader != nil {
		return c.checkReader(ctx, opts.Reader)
	}
	if c.dir == nil {
		return nil, errors.New("CheckIndex: a directory is required when no reader is provided")
	}
	return c.checkDirectory(ctx)
}

// checkDirectory loads the latest commit and inspects every segment in it.
// Each segment reader is opened independently so that one unreadable segment
// cannot prevent the other segments from being inspected.
func (c *CheckIndex) checkDirectory(ctx context.Context) (*IndexReport, error) {
	sis, err := ReadLatestCommit(ctx, c.dir)
	if err != nil {
		report := newIndexReport(0)
		report.Healthy = false
		report.Error = "no readable index commit found in directory"
		return report, err
	}

	opener := func(sci index.SegmentCommitInfo) (index.CodecReader, func(), error) {
		reader, openErr := NewSegmentReader(ctx, sci, sis.getIndexCreatedVersionMajor(), store.READ)
		if openErr != nil {
			return nil, nil, openErr
		}
		return reader, func() {
			_ = reader.Close()
		}, nil
	}

	return c.inspectCommit(ctx, sis.GetGeneration(), sis.AsList(), opener), nil
}

// checkReader inspects an already opened reader. The reader is never closed.
func (c *CheckIndex) checkReader(ctx context.Context, reader index.DirectoryReader) (*IndexReport, error) {
	if sdr, ok := reader.(*StandardDirectoryReader); ok {
		sis := sdr.GetSegmentInfos()
		leaves, err := reader.Leaves()
		if err != nil {
			report := newIndexReport(sis.GetGeneration())
			report.Healthy = false
			report.Error = "reader leaves are not accessible"
			return report, err
		}

		leafByName := make(map[string]index.CodecReader, len(leaves))
		for _, leaf := range leaves {
			if sr, ok := leaf.LeafReader().(*SegmentReader); ok {
				leafByName[sr.si.Info().Name()] = sr
			}
		}

		opener := func(sci index.SegmentCommitInfo) (index.CodecReader, func(), error) {
			if leaf, ok := leafByName[sci.Info().Name()]; ok {
				return leaf, func() {}, nil
			}
			return nil, nil, errors.New("segment reader is not available in the provided reader")
		}
		return c.inspectCommit(ctx, sis.GetGeneration(), sis.AsList(), opener), nil
	}

	// Generic fallback for foreign DirectoryReader implementations: rely only
	// on the public leaf view and preserve the reader-provided ordering.
	report, err := c.inspectGenericReader(ctx, reader)
	return report, err
}

// segmentReaderOpener opens (or reuses) the CodecReader for one segment.
// The release function must always be nil-safe and is invoked after the
// segment inspection.
type segmentReaderOpener func(sci index.SegmentCommitInfo) (index.CodecReader, func(), error)

// inspectCommit walks the segments declared by a commit in commit order and
// aggregates their reports. Cancellation is observed between segments and
// turns every not-yet-inspected segment into a deterministic CANCELLED entry.
func (c *CheckIndex) inspectCommit(ctx context.Context, generation int64,
	segments []index.SegmentCommitInfo, opener segmentReaderOpener) *IndexReport {

	report := newIndexReport(generation)
	report.SegmentCount = len(segments)

	docBase := 0
	for i, sci := range segments {
		if ctxErr := ctx.Err(); ctxErr != nil {
			for _, pending := range segments[i:] {
				report.Segments = append(report.Segments, cancelledSegmentReport(pending, docBase))
				if maxDoc, err := pending.Info().MaxDoc(); err == nil {
					docBase += maxDoc
				}
			}
			report.Healthy = false
			return report
		}

		segmentReport := c.inspectSegment(ctx, sci, docBase, opener)
		report.Segments = append(report.Segments, segmentReport)

		report.MaxDoc += segmentReport.MaxDoc
		report.NumDocs += segmentReport.LiveDocs
		report.TotalDeletedDocs += segmentReport.DeletedDocs
		report.TotalSoftDeletedDocs += segmentReport.SoftDeletedDocs
		if segmentReport.Status != SegmentStatusOK {
			report.Healthy = false
		}
		docBase += segmentReport.MaxDoc
	}

	return report
}

// inspectSegment runs every per-segment check. A failure of one check is
// recorded but never prevents the remaining checks or segments from running.
func (c *CheckIndex) inspectSegment(ctx context.Context, sci index.SegmentCommitInfo,
	docBase int, opener segmentReaderOpener) *SegmentReport {

	info := sci.Info()
	maxDoc, maxDocErr := info.MaxDoc()
	if maxDocErr != nil {
		maxDoc = 0
	}

	segmentReport := &SegmentReport{
		Name:            info.Name(),
		DocBase:         docBase,
		MaxDoc:          maxDoc,
		DeletedDocs:     sci.GetDelCount(),
		SoftDeletedDocs: sci.GetSoftDelCount(),
		Checks:          make([]SegmentCheckResult, 0, 3),
	}
	expectedLiveDocs := maxDoc - sci.GetDelCount()

	// The segment id check only needs committed metadata and therefore runs
	// even when the segment reader cannot be opened.
	segmentReport.addCheck(checkSegmentID(sci))

	reader, release, openErr := opener(sci)
	if release != nil {
		defer release()
	}

	// Reader accessibility is the gate for the live-docs inspection.
	accessResult, accessible := checkReaderAccess(ctx, sci, reader, openErr, maxDoc, expectedLiveDocs)
	segmentReport.addCheck(accessResult)

	if accessible {
		segmentReport.addCheck(checkLiveDocsCount(sci, reader, maxDoc, expectedLiveDocs))
		segmentReport.LiveDocs = countLiveDocs(reader, expectedLiveDocs)
	} else {
		segmentReport.addCheck(SegmentCheckResult{
			Name:    segmentCheckLiveDocs,
			Passed:  false,
			Message: fmt.Sprintf("segment %q: live docs cannot be inspected because the reader is not accessible", info.Name()),
		})
		// Fall back to the committed expectation so aggregate counts stay
		// deterministic even for an unreadable segment.
		segmentReport.LiveDocs = expectedLiveDocs
	}

	if ctx.Err() != nil {
		segmentReport.Status = SegmentStatusCancelled
		segmentReport.Error = fmt.Sprintf("segment %q: check cancelled", info.Name())
		return segmentReport
	}

	for _, check := range segmentReport.Checks {
		if !check.Passed {
			segmentReport.Status = SegmentStatusFailed
			segmentReport.Error = check.Message
			break
		}
	}
	return segmentReport
}

// inspectGenericReader handles DirectoryReader implementations that do not
// expose SegmentInfos. Only the public leaf API is used.
func (c *CheckIndex) inspectGenericReader(ctx context.Context, reader index.DirectoryReader) (*IndexReport, error) {
	leaves, err := reader.Leaves()
	if err != nil {
		report := newIndexReport(0)
		report.Healthy = false
		report.Error = "reader leaves are not accessible"
		return report, err
	}

	generation := int64(0)
	if dir := reader.Directory(); dir != nil {
		if sis, commitErr := ReadLatestCommit(ctx, dir); commitErr == nil {
			generation = sis.GetGeneration()
		}
	}

	report := newIndexReport(generation)
	report.SegmentCount = len(leaves)

	for ord, leaf := range leaves {
		if ctx.Err() != nil {
			report.Healthy = false
			break
		}

		lr := leaf.LeafReader()
		name := fmt.Sprintf("leaf-%d", ord)
		var codecReader index.CodecReader
		if sr, ok := lr.(*SegmentReader); ok {
			name = sr.si.Info().Name()
			codecReader = sr
		} else if cr, ok := lr.(index.CodecReader); ok {
			codecReader = cr
		}

		maxDoc := lr.MaxDoc()
		numDocs := lr.NumDocs()
		segmentReport := &SegmentReport{
			Name:        name,
			DocBase:     leaf.DocBase(),
			MaxDoc:      maxDoc,
			LiveDocs:    numDocs,
			DeletedDocs: maxDoc - numDocs,
			Checks:      make([]SegmentCheckResult, 0, 3),
		}
		segmentReport.addCheck(checkGenericSegmentID(name, codecReader))
		segmentReport.addCheck(checkGenericReaderAccess(name, codecReader, maxDoc, numDocs))
		segmentReport.addCheck(checkGenericLiveDocs(name, lr, maxDoc, numDocs))

		for _, check := range segmentReport.Checks {
			if !check.Passed {
				segmentReport.Status = SegmentStatusFailed
				segmentReport.Error = check.Message
				break
			}
		}
		if segmentReport.Status != SegmentStatusOK {
			report.Healthy = false
		}

		report.Segments = append(report.Segments, segmentReport)
		report.MaxDoc += maxDoc
		report.NumDocs += numDocs
		report.TotalDeletedDocs += maxDoc - numDocs
	}

	return report, nil
}

// addCheck records a check result and keeps the check ordering stable.
func (r *SegmentReport) addCheck(result SegmentCheckResult) {
	r.Checks = append(r.Checks, result)
}

// newIndexReport creates a report with a non-nil, ordered segment slice so
// JSON serialization is identical across repeated runs.
func newIndexReport(generation int64) *IndexReport {
	return &IndexReport{
		CommitGeneration: generation,
		Segments:         make([]*SegmentReport, 0),
		Healthy:          true,
	}
}

// cancelledSegmentReport builds the deterministic placeholder used for
// segments that were never inspected due to context cancellation.
func cancelledSegmentReport(sci index.SegmentCommitInfo, docBase int) *SegmentReport {
	info := sci.Info()
	maxDoc := 0
	if v, err := info.MaxDoc(); err == nil {
		maxDoc = v
	}
	deleted := sci.GetDelCount()
	return &SegmentReport{
		Name:            info.Name(),
		DocBase:         docBase,
		MaxDoc:          maxDoc,
		LiveDocs:        maxDoc - deleted,
		DeletedDocs:     deleted,
		SoftDeletedDocs: sci.GetSoftDelCount(),
		Status:          SegmentStatusCancelled,
		Error:           fmt.Sprintf("segment %q: check cancelled", info.Name()),
		Checks:          []SegmentCheckResult{},
	}
}

// checkSegmentID verifies that the committed segment identifier and the
// segment-commit identifier have the expected fixed length. Identifier bytes
// are never exposed.
func checkSegmentID(sci index.SegmentCommitInfo) SegmentCheckResult {
	name := sci.Info().Name()

	segmentID := sci.Info().GetID()
	if len(segmentID) != ID_LENGTH {
		return SegmentCheckResult{
			Name:    segmentCheckSegmentID,
			Passed:  false,
			Message: fmt.Sprintf("segment %q: segment id length is %d, expected %d", name, len(segmentID), ID_LENGTH),
		}
	}

	commitID := sci.GetId()
	if commitID != nil && len(commitID) != ID_LENGTH {
		return SegmentCheckResult{
			Name:    segmentCheckSegmentID,
			Passed:  false,
			Message: fmt.Sprintf("segment %q: segment commit id length is %d, expected %d", name, len(commitID), ID_LENGTH),
		}
	}

	return SegmentCheckResult{Name: segmentCheckSegmentID, Passed: true}
}

// checkReaderAccess verifies that the segment reader opens and exposes
// internally consistent, readable metadata.
func checkReaderAccess(ctx context.Context, sci index.SegmentCommitInfo,
	reader index.CodecReader, openErr error, maxDoc, expectedLiveDocs int) (SegmentCheckResult, bool) {

	name := sci.Info().Name()
	if openErr != nil || reader == nil {
		return SegmentCheckResult{
			Name:    segmentCheckReaderAccess,
			Passed:  false,
			Message: fmt.Sprintf("segment %q: segment reader cannot be opened", name),
		}, false
	}

	if reader.MaxDoc() != maxDoc {
		return SegmentCheckResult{
			Name:    segmentCheckReaderAccess,
			Passed:  false,
			Message: fmt.Sprintf("segment %q: reader maxDoc is %d, expected %d", name, reader.MaxDoc(), maxDoc),
		}, false
	}

	if reader.NumDocs() != expectedLiveDocs {
		return SegmentCheckResult{
			Name:    segmentCheckReaderAccess,
			Passed:  false,
			Message: fmt.Sprintf("segment %q: reader numDocs is %d, expected %d", name, reader.NumDocs(), expectedLiveDocs),
		}, false
	}

	if reader.GetFieldInfos() == nil {
		return SegmentCheckResult{
			Name:    segmentCheckReaderAccess,
			Passed:  false,
			Message: fmt.Sprintf("segment %q: field infos are not accessible", name),
		}, false
	}

	// Probe stored-field accessibility on the first document. An empty segment
	// has nothing to probe and is considered accessible at this stage.
	if maxDoc > 0 {
		if _, err := reader.Document(ctx, 0); err != nil {
			if ctx.Err() != nil {
				return SegmentCheckResult{
					Name:    segmentCheckReaderAccess,
					Passed:  false,
					Message: fmt.Sprintf("segment %q: check cancelled while reading stored fields", name),
				}, false
			}
			return SegmentCheckResult{
				Name:    segmentCheckReaderAccess,
				Passed:  false,
				Message: fmt.Sprintf("segment %q: stored fields are not accessible", name),
			}, false
		}
	}

	return SegmentCheckResult{Name: segmentCheckReaderAccess, Passed: true}, true
}

// checkLiveDocsCount verifies the live-docs Bits against the committed
// deletion count, mirroring the semantics of TestLiveDocs.
func checkLiveDocsCount(sci index.SegmentCommitInfo, reader index.CodecReader,
	maxDoc, expectedLiveDocs int) SegmentCheckResult {

	name := sci.Info().Name()
	bits := reader.GetLiveDocs()

	if sci.HasDeletions() {
		if bits == nil {
			return SegmentCheckResult{
				Name:    segmentCheckLiveDocs,
				Passed:  false,
				Message: fmt.Sprintf("segment %q: deletions are committed but live docs bits are missing", name),
			}
		}
		if int(bits.Len()) != maxDoc {
			return SegmentCheckResult{
				Name:    segmentCheckLiveDocs,
				Passed:  false,
				Message: fmt.Sprintf("segment %q: live docs length is %d, expected %d", name, bits.Len(), maxDoc),
			}
		}
		live := popCount(bits)
		if live != expectedLiveDocs {
			return SegmentCheckResult{
				Name:    segmentCheckLiveDocs,
				Passed:  false,
				Message: fmt.Sprintf("segment %q: live docs count is %d, expected %d", name, live, expectedLiveDocs),
			}
		}
		return SegmentCheckResult{Name: segmentCheckLiveDocs, Passed: true}
	}

	if bits != nil {
		if int(bits.Len()) != maxDoc {
			return SegmentCheckResult{
				Name:    segmentCheckLiveDocs,
				Passed:  false,
				Message: fmt.Sprintf("segment %q: live docs length is %d, expected %d", name, bits.Len(), maxDoc),
			}
		}
		for i := 0; i < maxDoc; i++ {
			if !bits.Test(uint(i)) {
				return SegmentCheckResult{
					Name:    segmentCheckLiveDocs,
					Passed:  false,
					Message: fmt.Sprintf("segment %q: live docs mark document %d as deleted although no deletions are committed", name, i),
				}
			}
		}
	}
	return SegmentCheckResult{Name: segmentCheckLiveDocs, Passed: true}
}

// countLiveDocs returns the number of set bits, or the expected count when the
// segment has no deletions and therefore exposes no live-docs Bits.
func countLiveDocs(reader index.CodecReader, expectedLiveDocs int) int {
	bits := reader.GetLiveDocs()
	if bits == nil {
		return expectedLiveDocs
	}
	return popCount(bits)
}

// popCount counts the set bits of a Bits instance.
func popCount(bits interface {
	Len() uint
	Test(index uint) bool
}) int {
	count := 0
	for i, size := 0, int(bits.Len()); i < size; i++ {
		if bits.Test(uint(i)) {
			count++
		}
	}
	return count
}

// checkGenericSegmentID is the best-effort id check for foreign readers.
func checkGenericSegmentID(name string, reader index.CodecReader) SegmentCheckResult {
	sr, ok := reader.(*SegmentReader)
	if !ok {
		return SegmentCheckResult{
			Name:    segmentCheckSegmentID,
			Passed:  true,
			Message: fmt.Sprintf("segment %q: segment id is not exposed by the reader wrapper", name),
		}
	}
	return checkSegmentID(sr.si)
}

// checkGenericReaderAccess validates a leaf exposed by a foreign reader.
func checkGenericReaderAccess(name string, reader index.CodecReader, maxDoc, numDocs int) SegmentCheckResult {
	if reader == nil {
		return SegmentCheckResult{
			Name:    segmentCheckReaderAccess,
			Passed:  false,
			Message: fmt.Sprintf("segment %q: segment reader cannot be opened", name),
		}
	}
	if reader.MaxDoc() != maxDoc || reader.NumDocs() != numDocs || reader.GetFieldInfos() == nil {
		return SegmentCheckResult{
			Name:    segmentCheckReaderAccess,
			Passed:  false,
			Message: fmt.Sprintf("segment %q: reader metadata is inconsistent or inaccessible", name),
		}
	}
	return SegmentCheckResult{Name: segmentCheckReaderAccess, Passed: true}
}

// checkGenericLiveDocs validates live docs for a foreign reader leaf.
func checkGenericLiveDocs(name string, reader index.LeafReader, maxDoc, numDocs int) SegmentCheckResult {
	bits := reader.GetLiveDocs()
	if bits == nil {
		if maxDoc != numDocs {
			return SegmentCheckResult{
				Name:    segmentCheckLiveDocs,
				Passed:  false,
				Message: fmt.Sprintf("segment %q: deletions exist but live docs bits are missing", name),
			}
		}
		return SegmentCheckResult{Name: segmentCheckLiveDocs, Passed: true}
	}
	if int(bits.Len()) != maxDoc {
		return SegmentCheckResult{
			Name:    segmentCheckLiveDocs,
			Passed:  false,
			Message: fmt.Sprintf("segment %q: live docs length is %d, expected %d", name, bits.Len(), maxDoc),
		}
	}
	live := popCount(bits)
	if live != numDocs {
		return SegmentCheckResult{
			Name:    segmentCheckLiveDocs,
			Passed:  false,
			Message: fmt.Sprintf("segment %q: live docs count is %d, expected %d", name, live, numDocs),
		}
	}
	return SegmentCheckResult{Name: segmentCheckLiveDocs, Passed: true}
}
