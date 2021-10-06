#ifndef POKE_HEADER_MATRIX_H_
#define POKE_HEADER_MATRIX_H_
#include <algorithm>
#include <array>
#include <exception>
#include <numeric>
#include <type_traits>

namespace poke {
namespace matrix {

template <typename T, size_t Row, size_t Col>
class Matrix;

namespace detail {
/**
 * A Holder holds the copy of the item if its size < 64
 * Else it holds the const reference
 */
template <typename T>
struct Holder {
  explicit Holder(T value) : storage_(std::move(value)) {}
  const T *operator->() const { return &storage_; }
  T *operator->() { return &storage_; }
  const T &operator*() const { return storage_; }
  T &operator*() { return storage_; }
  T storage_;
};

template <typename T, size_t Row, size_t Col>
struct Holder<Matrix<T, Row, Col>> {
  explicit Holder(Matrix<T, Row, Col> &value) : storage_(value) {}
  const Matrix<T, Row, Col> *operator->() const { return &storage_; }
  Matrix<T, Row, Col> *operator->() { return &storage_; }
  const Matrix<T, Row, Col> &operator*() const { return storage_; }
  Matrix<T, Row, Col> &operator*() { return storage_; }
  Matrix<T, Row, Col> &storage_;
};

template <typename T, size_t Row, size_t Col>
struct Holder<const Matrix<T, Row, Col>> {
  explicit Holder(const Matrix<T, Row, Col> &value) : storage_(value) {}
  const Matrix<T, Row, Col> *operator->() const { return &storage_; }
  const Matrix<T, Row, Col> *operator->() { return &storage_; }
  const Matrix<T, Row, Col> &operator*() const { return storage_; }
  const Matrix<T, Row, Col> &operator*() { return storage_; }
  const Matrix<T, Row, Col> &storage_;
};

template <typename Base>
class MatrixIterator {
 public:
  explicit MatrixIterator(Base &base, size_t idx) noexcept : base_(base), idx_(idx) {}
  decltype(auto) operator*() noexcept {
    constexpr size_t cols = Base::Cols();
    size_t col = idx_ % cols, row = idx_ / cols;
    return base_(row, col);
  }
  decltype(auto) operator*() const noexcept {
    constexpr size_t cols = Base::Cols();
    size_t col = idx_ % cols, row = idx_ / cols;
    return base_(row, col);
  }
  MatrixIterator &operator++() {
    ++idx_;
    return *this;
  }
  MatrixIterator operator++(int) {
    MatrixIterator<Base> res(base_, idx_++);
  }
  MatrixIterator &operator--() {
    --idx_;
    return *this;
  }
  MatrixIterator operator--(int) {
    return MatrixIterator<Base>(base_, idx_--);
  }
  size_t operator-(const MatrixIterator &other) const { return idx_ - other.idx_; }
  bool operator==(const MatrixIterator<Base> &other) const noexcept { return idx_ == other.idx_; }
  bool operator!=(const MatrixIterator<Base> &other) const noexcept { return idx_ != other.idx_; }
 private:
  Base &base_;
  size_t idx_;
};

template <size_t Row, size_t Col>
void check_border(size_t row, size_t col) {
  const bool condition = row >= 0 && row < Row && col >= 0 && col < Col;
  if (!condition) {
    throw std::range_error("row/col out of range");
  }
}

template <typename Mat, typename Func, typename Seq>
struct aggregate_trait;
template <typename Mat, typename Func, size_t...Is>
struct aggregate_trait<Mat, Func, std::index_sequence<Is...>> {
  static auto agg(const Mat &mat, const Func &func) {
    constexpr size_t Cols = Mat::Cols();
    return std::invoke(func, mat(Is / Cols, Is % Cols)...);
  }
};
} // namespace detail

enum Dimension {
  Row = 0,
  Col = 1
};

template <typename Base, Dimension dim>
class Aggregate {
 public:
  explicit Aggregate(const Base &base) : holder_(base) {}
  template <typename Func>
  auto Apply(Func &&func) const {
    constexpr size_t Rows = dim == Dimension::Row ? Base::Rows() : 1;
    constexpr size_t Cols = dim == Dimension::Col ? Base::Cols() : 1;
    Matrix<typename Base::value_type, Rows, Cols> mat;
    constexpr size_t n = Rows > Cols ? Rows : Cols;
    constexpr size_t row_size = dim == Dimension::Row ? 1 : Base::Rows();
    constexpr size_t col_size = dim == Dimension::Col ? 1 : Base::Cols();
    for (size_t i = 0; i < n; i++) {
      size_t row_id = dim == Dimension::Row ? i : 0;
      size_t col_id = dim == Dimension::Col ? i : 0;
      mat(row_id, col_id) = holder_->template Block<row_size, col_size>(row_id, col_id).Reduce(std::forward<Func>(func));
    }
    return mat;
  }
  auto Sum() const {
    return Apply([](auto...items) { return (items + ...); });
  }
  auto All() const {
    return Apply([](auto...items) { return (items && ...); });
  }
  auto Any() const {
    return Apply([](auto...items) { return (items || ...); });
  }
 private:
  detail::Holder<const Base> holder_;
};

template <typename Base, Dimension dim, size_t Window>
class RollingOperation {
 public:
  explicit RollingOperation(const Base &base) : holder_(base) {}
  template <typename Func>
  auto Apply(Func &&func) const {
    static_assert(dim == Dimension::Row ? Base::Rows() >= Window : Base::Cols() >= Window);
    constexpr size_t Rows = dim == Dimension::Row ? Base::Rows() - Window + 1 : Base::Rows();
    constexpr size_t Cols = dim == Dimension::Col ? Base::Cols() - Window + 1 : Base::Cols();
    Matrix<typename Base::value_type, Rows, Cols> mat;
    constexpr size_t n = Rows > Cols ? Rows : Cols;
    constexpr size_t row_size = dim == Dimension::Row ? Window : 1;
    constexpr size_t col_size = dim == Dimension::Col ? Window : 1;
    for (size_t r = 0; r < Rows; r++) {
      for (size_t c = 0; c < Cols; c++) {
        mat(r, c) = holder_->template Block<row_size, col_size>(r, c).Reduce(std::forward<Func>(func));
      }
    }
    return mat;
  }
  auto Sum() const {
    return Apply([](auto...items) { return (items + ...); });
  }
  auto All() const {
    return Apply([](auto...items) { return (items && ...); });
  }
  auto Any() const {
    return Apply([](auto...items) { return (items || ...); });
  }
 private:
  detail::Holder<const Base> holder_;
};

template <typename, size_t, size_t>
class MatrixBlock;

template <typename Derived>
class MatrixBase {
 public:
//  using value_type = std::remove_reference_t<decltype(std::declval<Derived>().GetValue(0, 0))>;
  static constexpr size_t Rows() noexcept { return Derived::Rows(); }
  static constexpr size_t Cols() noexcept { return Derived::Cols(); }
  auto begin() noexcept { return detail::MatrixIterator<MatrixBase<Derived>>(*this, 0); }
  auto begin() const noexcept { return detail::MatrixIterator<const MatrixBase<Derived>>(*this, 0); }
  auto end() noexcept { return detail::MatrixIterator<MatrixBase<Derived>>(*this, Rows() * Cols()); }
  auto end() const noexcept { return detail::MatrixIterator<const MatrixBase<Derived>>(*this, Rows() * Cols()); }
  decltype(auto) operator() (size_t row, size_t col) {
    detail::check_border<Rows(), Cols()>(row, col);
    return static_cast<Derived *>(this)->GetValue(row, col);
  }
  decltype(auto) operator() (size_t row, size_t col) const {
    detail::check_border<Rows(), Cols()>(row, col);
    return static_cast<const Derived *>(this)->GetValue(row, col);
  }
  template <size_t Row, size_t Col>
  auto Block(size_t row_begin, size_t col_begin) {
    return MatrixBlock<Derived, Row, Col>(*static_cast<Derived *>(this), row_begin, col_begin);
  }
  template <size_t Row, size_t Col>
  auto Block(size_t row_begin, size_t col_begin) const {
    return MatrixBlock<const Derived, Row, Col>(*static_cast<const Derived *>(this), row_begin, col_begin);
  }
  template <typename Func>
  auto Reduce(Func &&func) const {
    using Seq = std::make_index_sequence<Rows() * Cols()>;
    return detail::aggregate_trait<MatrixBase<Derived>, Func, Seq>::agg(*this, std::forward<Func>(func));
  }
  template <Dimension dim>
  auto ReduceBy() const {
    return Aggregate<const Derived, dim>(*static_cast<const Derived *>(this));
  }
  template <Dimension dim, size_t Window>
  auto RollingBy() {
    return RollingOperation<Derived, dim, Window>(*static_cast<Derived *>(this));
  }
  template <Dimension dim, size_t Window>
  auto RollingBy() const {
    return RollingOperation<const Derived, dim, Window>(*static_cast<const Derived *>(this));
  }
  template <typename UnaryOp>
  auto Map(UnaryOp &&func) const {
    using source_type = std::remove_reference_t<decltype(std::declval<MatrixBase<Derived>>()(0, 0))>;
    using target_type = std::remove_cvref_t<std::invoke_result_t<UnaryOp, source_type>>;
    Matrix<target_type, Rows(), Cols()> mat;
    std::transform(begin(), end(), mat.begin(), std::forward<UnaryOp>(func));
    return mat;
}
};

template<typename T, size_t Row, size_t Col>
class Matrix : public MatrixBase<Matrix<T, Row, Col>> {
  static_assert(Row > 0, "row > 0 violated");
  static_assert(Col > 0, "col > 0 violated");
 public:
  using value_type = T;
  using MatrixBase<Matrix<T, Row, Col>>::operator();
  using MatrixBase<Matrix<T, Row, Col>>::begin;
  using MatrixBase<Matrix<T, Row, Col>>::end;
  static constexpr size_t Rows() noexcept { return Row; }
  static constexpr size_t Cols() noexcept { return Col; }
  T &GetValue(size_t row, size_t col) { return data_[row * Col + col]; }
  const T &GetValue(size_t row, size_t col) const { return data_[row * Col + col]; }

  static Matrix Empty() noexcept { return Matrix(); }
  static Matrix Zeros() noexcept {
    Matrix mat;
    std::fill(mat.begin(), mat.end(), T(0));
    return mat;
  }
  static Matrix Ones() noexcept {
    Matrix mat;
    std::fill(mat.begin(), mat.end(), T(1));
    return mat;
  }
 protected:
  constexpr Matrix() noexcept = default;
  template <typename>
  friend class MatrixBase;
  template <typename, Dimension>
  friend class Aggregate;
  template <typename, Dimension, size_t>
  friend class RollingOperation;
 private:
  std::array<T, Row * Col> data_;
};

template <typename Base, size_t Row, size_t Col>
class MatrixBlock : public MatrixBase<MatrixBlock<Base, Row, Col>> {
  static_assert(Row << Base::Rows(), "Block rows must be less than matrix rows");
  static_assert(Col << Base::Cols(), "Block cols must be less than matrix cols");
 public:
  using value_type = typename Base::value_type;
  using MatrixBase<MatrixBlock<Base, Row, Col>>::operator();
  using MatrixBase<MatrixBlock<Base, Row, Col>>::begin;
  using MatrixBase<MatrixBlock<Base, Row, Col>>::end;
  MatrixBlock(Base &mat, size_t row_begin, size_t col_begin) : row_begin_(row_begin), col_begin_(col_begin), holder_(mat) {}
  static constexpr size_t Rows() { return Row; }
  static constexpr size_t Cols() { return Col; }
  value_type &GetValue(size_t row, size_t col) { return holder_->operator()(row + row_begin_, col + col_begin_); }
  const value_type &GetValue(size_t row, size_t col) const { return holder_->operator()(row + row_begin_, col + col_begin_); }
 private:
  const size_t row_begin_, col_begin_;
  detail::Holder<Base> holder_;
};

template <typename Base>
std::ostream &operator<<(std::ostream &os, const MatrixBase<Base> &mat) {
  for (size_t r = 0; r < mat.Rows(); r++) {
    for (size_t c = 0; c < mat.Cols(); c++) {
      os << mat(r, c) << (c == mat.Cols() - 1 ? '\n' : ',');
    }
  }
  return os;
}

} // namespace matrix
} // namespace poke

namespace std {
template <typename Base>
struct iterator_traits<poke::matrix::detail::MatrixIterator<Base>> {
  using difference_type = size_t;
  using value_type = std::remove_cvref_t<decltype(std::declval<Base>()(0, 0))>;
  using reference = value_type&;
  using iterator_category = std::forward_iterator_tag;
};
} // namespac std
#endif //POKE_HEADER_MATRIX_H_
